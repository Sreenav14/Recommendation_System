import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

TRAIN_HIST_PATH = "../data/processed/train_hist.parquet"
VAL_HIST_PATH   = "../data/processed/val_hist.parquet"
MOVIES_PATH     = "../data/processed/movies_clean.parquet"

OUT_DIR = "../artifact/tfrs_retrival_model_history"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 4096
EMBED_DIM = 64
EPOCHS = 5
MAX_HIST = 50
PAD_TOKEN = "[PAD]"


def pad_or_trim(hist, max_len=50):
    """Keep last max_len. Right-pad with PAD_TOKEN."""
    hist = [str(x) for x in hist]
    if len(hist) > max_len:
        hist = hist[-max_len:]
    if len(hist) < max_len:
        hist = hist + [PAD_TOKEN] * (max_len - len(hist))
    return hist


def df_to_tfds_hist_fixed(path: str) -> tf.data.Dataset:
    """
    Returns dataset where hist_movieIds is a FIXED length (MAX_HIST,) string tensor.
    """
    df = pd.read_parquet(path, columns=["userId", "movieId", "hist_movieIds"])
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)

    # ensure python lists, then pad
    fixed_hists = []
    for h in df["hist_movieIds"].values:
        if isinstance(h, np.ndarray):
            h = h.tolist()
        fixed_hists.append(pad_or_trim(h, MAX_HIST))

    ds = tf.data.Dataset.from_tensor_slices({
        "userId": df["userId"].values,
        "movieId": df["movieId"].values,
        "hist_movieIds": np.array(fixed_hists, dtype=object),  # (N, 50) object array
    })

    # convert hist_movieIds element into tf.string tensor shape (50,)
    ds = ds.map(lambda x: {
        "userId": tf.cast(x["userId"], tf.string),
        "movieId": tf.cast(x["movieId"], tf.string),
        "hist_movieIds": tf.cast(x["hist_movieIds"], tf.string),
    }, num_parallel_calls=tf.data.AUTOTUNE)

    return ds


def build_movie_lookup_from_movies(movies_path: str):
    movies_df = pd.read_parquet(movies_path, columns=["movieId"])
    movies_df["movieId"] = movies_df["movieId"].astype(str)

    candidate_ids = movies_df["movieId"].values

    # IMPORTANT: include PAD_TOKEN in vocab via mask_token
    lookup = tf.keras.layers.StringLookup(
        mask_token=PAD_TOKEN,   # becomes id=0
        oov_token="[OOV]",
        output_mode="int"
    )

    # adapt on candidates (PAD handled automatically as mask_token)
    lookup.adapt(tf.data.Dataset.from_tensor_slices(candidate_ids).batch(100_000))
    return lookup


class TwoTowerHistory(tfrs.models.Model):
    """
    User tower: Embedding(hist_ids, mask_zero=True) -> GlobalAvgPool(masked) -> MLP
    Movie tower: Embedding(movie_id) -> MLP
    """
    def __init__(self, movie_lookup: tf.keras.layers.StringLookup):
        super().__init__()
        self.movie_lookup = movie_lookup

        self.embedding = tf.keras.layers.Embedding(
            input_dim=movie_lookup.vocabulary_size(),
            output_dim=EMBED_DIM,
            mask_zero=True  # <-- KEY: ignores PAD id=0
        )

        self.hist_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.user_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBED_DIM),
        ])

        self.movie_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBED_DIM),
        ])

        self.task = tfrs.tasks.Retrieval()

    def user_embed_from_history(self, hist_movie_ids: tf.Tensor) -> tf.Tensor:
        # hist_movie_ids: (batch, 50) strings
        hist_int = self.movie_lookup(hist_movie_ids)      # (batch, 50) ints, PAD -> 0
        hist_vecs = self.embedding(hist_int)              # (batch, 50, D) + mask

        user_vec = self.hist_pool(hist_vecs)              # respects mask
        return self.user_mlp(user_vec)

    def movie_embed(self, movie_ids: tf.Tensor) -> tf.Tensor:
        mid = self.movie_lookup(movie_ids)
        mvec = self.embedding(mid)                        # (batch, D)
        return self.movie_mlp(mvec)

    def compute_loss(self, features, training=False):
        user_vec = self.user_embed_from_history(features["hist_movieIds"])
        movie_vec = self.movie_embed(features["movieId"])
        return self.task(user_vec, movie_vec)


def main():
    print("Loading datasets (fixed length histories)...")
    train_raw = df_to_tfds_hist_fixed(TRAIN_HIST_PATH)
    val_raw   = df_to_tfds_hist_fixed(VAL_HIST_PATH)

    print("Building movie vocab...")
    movie_lookup = build_movie_lookup_from_movies(MOVIES_PATH)

    train_ds = (
        train_raw
        .shuffle(200_000, seed=42, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_raw
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = TwoTowerHistory(movie_lookup)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    print("Training history-based two-tower with PAD masking...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # export inference models (dense input: (None, 50))
    infer_dir = os.path.join(OUT_DIR, "inference")
    os.makedirs(infer_dir, exist_ok=True)

    hist_in = tf.keras.Input(shape=(MAX_HIST,), dtype=tf.string, name="hist_movieIds")
    u = model.user_embed_from_history(hist_in)
    u = tf.math.l2_normalize(u, axis=1)
    user_model = tf.keras.Model(hist_in, u, name="user_embedder")

    movie_in = tf.keras.Input(shape=(), dtype=tf.string, name="movieId")
    m = model.movie_embed(movie_in)
    m = tf.math.l2_normalize(m, axis=1)
    movie_model = tf.keras.Model(movie_in, m, name="movie_embedder")

    tf.saved_model.save(user_model, os.path.join(infer_dir, "user_embedder"))
    tf.saved_model.save(movie_model, os.path.join(infer_dir, "movie_embedder"))

    # meta for evaluation scripts
    meta = {"max_hist": MAX_HIST, "pad_token": PAD_TOKEN}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history.history, f)

    print("\n=== History model trained + exported ===")
    print("Inference dir:", infer_dir)
    print("Meta:", os.path.join(OUT_DIR, "meta.json"))


if __name__ == "__main__":
    main()
