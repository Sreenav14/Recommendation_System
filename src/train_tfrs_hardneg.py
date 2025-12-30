import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

TRAIN_PATH = "../data/processed/train.parquet"
VAL_PATH = "../data/processed/val.parquet"

OUT_DIR = "../artifact/tfrs_retrival_model_hardneg"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 4096
EMBED_DIM = 64
EPOCHS = 5

NEG_POOL_SIZE = 5000  # top popular movies to sample negatives from


def build_lookup(values):
    lookup = tf.keras.layers.StringLookup(mask_token=None, output_mode="int")
    lookup.adapt(tf.data.Dataset.from_tensor_slices(values).batch(100_000))
    return lookup


class TwoTowerBPR(tf.keras.Model):
    def __init__(self, user_lookup, movie_lookup):
        super().__init__()
        self.user_lookup = user_lookup
        self.movie_lookup = movie_lookup

        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(user_lookup.vocabulary_size(), EMBED_DIM),
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBED_DIM),
        ])

        self.movie_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(movie_lookup.vocabulary_size(), EMBED_DIM),
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBED_DIM),
        ])

    def call(self, inputs):
        # not used directly
        return inputs

    def embed_user(self, user_ids_str):
        u = self.user_tower(self.user_lookup(user_ids_str))
        return tf.math.l2_normalize(u, axis=1)

    def embed_movie(self, movie_ids_str):
        v = self.movie_tower(self.movie_lookup(movie_ids_str))
        return tf.math.l2_normalize(v, axis=1)


def make_train_ds(train_df, neg_pool):
    # Build dataset of (user, pos_movie, neg_movie)
    users = train_df["userId"].values
    pos = train_df["movieId"].values

    rng = np.random.default_rng(42)
    neg = rng.choice(neg_pool, size=len(train_df), replace=True)

    ds = tf.data.Dataset.from_tensor_slices({
        "userId": users,
        "pos_movieId": pos,
        "neg_movieId": neg,
    })
    return ds


def bpr_loss(u, v_pos, v_neg):
    # maximize u·v_pos > u·v_neg
    pos_scores = tf.reduce_sum(u * v_pos, axis=1)
    neg_scores = tf.reduce_sum(u * v_neg, axis=1)
    return tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))


def main():
    train_df = pd.read_parquet(TRAIN_PATH, columns=["userId", "movieId"])
    val_df = pd.read_parquet(VAL_PATH, columns=["userId", "movieId"])

    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)
    val_df["userId"] = val_df["userId"].astype(str)
    val_df["movieId"] = val_df["movieId"].astype(str)

    # Popular movies pool for hard negatives
    pop_counts = train_df["movieId"].value_counts()
    neg_pool = pop_counts.index.tolist()[:NEG_POOL_SIZE]
    neg_pool = np.array(neg_pool, dtype=object)

    # Lookups
    user_lookup = build_lookup(train_df["userId"].unique())
    movie_lookup = build_lookup(pd.concat([train_df["movieId"], val_df["movieId"]]).unique())

    model = TwoTowerBPR(user_lookup, movie_lookup)
    opt = tf.keras.optimizers.Adagrad(learning_rate=0.05)

    train_ds = (
        make_train_ds(train_df, neg_pool)
        .shuffle(200_000, seed=42, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # simple val sampled metric: average pos score - random neg score
    val_users = tf.constant(val_df["userId"].values[:20000], dtype=tf.string)
    val_pos = tf.constant(val_df["movieId"].values[:20000], dtype=tf.string)
    rng = np.random.default_rng(7)
    val_neg = tf.constant(rng.choice(neg_pool, size=len(val_users)), dtype=tf.string)

    for epoch in range(EPOCHS):
        losses = []
        for batch in train_ds:
            with tf.GradientTape() as tape:
                u = model.embed_user(tf.constant(batch["userId"], dtype=tf.string))
                vp = model.embed_movie(tf.constant(batch["pos_movieId"], dtype=tf.string))
                vn = model.embed_movie(tf.constant(batch["neg_movieId"], dtype=tf.string))
                loss = bpr_loss(u, vp, vn)

            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(loss.numpy())

        # quick val signal
        u = model.embed_user(val_users)
        vp = model.embed_movie(val_pos)
        vn = model.embed_movie(val_neg)
        pos_scores = tf.reduce_sum(u * vp, axis=1)
        neg_scores = tf.reduce_sum(u * vn, axis=1)
        margin = float(tf.reduce_mean(pos_scores - neg_scores).numpy())

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {np.mean(losses):.4f} - val_margin: {margin:.4f}")

    # Export inference models
    infer_dir = os.path.join(OUT_DIR, "inference")
    os.makedirs(infer_dir, exist_ok=True)

    user_in = tf.keras.Input(shape=(), dtype=tf.string, name="userId")
    movie_in = tf.keras.Input(shape=(), dtype=tf.string, name="movieId")

    u = model.embed_user(user_in)
    m = model.embed_movie(movie_in)

    user_model = tf.keras.Model(user_in, u, name="user_embedder")
    movie_model = tf.keras.Model(movie_in, m, name="movie_embedder")

    tf.saved_model.save(user_model, os.path.join(infer_dir, "user_embedder"))
    tf.saved_model.save(movie_model, os.path.join(infer_dir, "movie_embedder"))

    print("Saved inference models to:", infer_dir)

if __name__ == "__main__":
    main()
