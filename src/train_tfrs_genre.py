import os
import json
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

TRAIN_PATH = "../data/processed/train.parquet"
VAL_PATH = "../data/processed/val.parquet"
MOVIES_PATH = "../data/processed/movies_clean.parquet"

OUT_DIR = "../artifact/tfrs_retrival_model_genre"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 4096
EMBED_DIM = 64
EPOCHS = 5  # start with 5


def load_movies_map():
    movies = pd.read_parquet(MOVIES_PATH, columns=["movieId", "genres"])
    movies["movieId"] = movies["movieId"].astype(str)
    movies["genres"] = movies["genres"].fillna("")
    return dict(zip(movies["movieId"], movies["genres"]))


def df_to_tfds(path: str, movieid_to_genres: dict) -> tf.data.Dataset:
    df = pd.read_parquet(path, columns=["userId", "movieId"])
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)

    # Add genres column using pandas map (fast + simple)
    df["genres"] = df["movieId"].map(movieid_to_genres).fillna("")

    ds = tf.data.Dataset.from_tensor_slices({
        "userId": df["userId"].values,
        "movieId": df["movieId"].values,
        "genres": df["genres"].values,
    })
    return ds


def build_string_vocab(ds: tf.data.Dataset, key: str):
    vocab_ds = ds.map(lambda x: x[key])
    lookup = tf.keras.layers.StringLookup(mask_token=None)
    lookup.adapt(vocab_ds.batch(100_000))
    return lookup


class TwoTowerGenres(tfrs.models.Model):
    def __init__(self, user_lookup, movie_lookup, genre_vectorizer):
        super().__init__()
        self.user_lookup = user_lookup
        self.movie_lookup = movie_lookup
        self.genre_vectorizer = genre_vectorizer

        # User tower
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(user_lookup.vocabulary_size(), EMBED_DIM),
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBED_DIM),
        ])

        # MovieId embedding
        self.movie_id_emb = tf.keras.layers.Embedding(movie_lookup.vocabulary_size(), EMBED_DIM)

        # Movie MLP after concatenating [movieId_emb, genre_multi_hot]
        self.movie_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBED_DIM),
        ])

        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        # user
        user_ids = self.user_lookup(features["userId"])
        user_vec = self.user_tower(user_ids)

        # movie id embedding
        movie_ids = self.movie_lookup(features["movieId"])
        movie_id_vec = self.movie_id_emb(movie_ids)  # (batch, EMBED_DIM)

        # genres multi-hot (batch, num_genres)
        genre_vec = tf.cast(self.genre_vectorizer(features["genres"]), tf.float32)

        # combine
        combined = tf.concat([movie_id_vec, genre_vec], axis=1)
        movie_vec = self.movie_mlp(combined)

        return self.task(user_vec, movie_vec)


def main():
    # Load mapping movieId -> genres string
    movieid_to_genres = load_movies_map()

    # Build datasets with genres attached
    train_raw = df_to_tfds(TRAIN_PATH, movieid_to_genres)
    val_raw = df_to_tfds(VAL_PATH, movieid_to_genres)

    # Build vocab for userId and movieId
    user_lookup = build_string_vocab(train_raw, "userId")
    movie_lookup = build_string_vocab(train_raw, "movieId")

    # Build genre vocabulary and a vectorizer (multi-hot)
    # We split genres like "Action|Comedy" into tokens using "|" delimiter.
    # We do it with TextVectorization (works well).
    genre_text = train_raw.map(lambda x: tf.strings.regex_replace(x["genres"], r"\|", " "))
    genre_vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        output_mode="multi_hot",
    )
    genre_vectorizer.adapt(genre_text.batch(100_000))

    train_ds = (
        train_raw.shuffle(200_000, seed=42, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = val_raw.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    model = TwoTowerGenres(user_lookup, movie_lookup, genre_vectorizer)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Export inference models
    infer_dir = os.path.join(OUT_DIR, "inference")
    os.makedirs(infer_dir, exist_ok=True)

    # user embedder
    user_in = tf.keras.Input(shape=(), dtype=tf.string, name="userId")
    u = model.user_tower(model.user_lookup(user_in))
    u = tf.math.l2_normalize(u, axis=1)
    user_model = tf.keras.Model(user_in, u, name="user_embedder")

    # movie embedder
    movie_in = tf.keras.Input(shape=(), dtype=tf.string, name="movieId")
    genres_in = tf.keras.Input(shape=(), dtype=tf.string, name="genres")

    mid = model.movie_lookup(movie_in)
    mid_vec = model.movie_id_emb(mid)
    gvec = tf.cast(model.genre_vectorizer(tf.strings.regex_replace(genres_in, r"\|", " ")), tf.float32)

    m = model.movie_mlp(tf.concat([mid_vec, gvec], axis=1))
    m = tf.math.l2_normalize(m, axis=1)
    movie_model = tf.keras.Model([movie_in, genres_in], m, name="movie_embedder")

    tf.saved_model.save(user_model, os.path.join(infer_dir, "user_embedder"))
    tf.saved_model.save(movie_model, os.path.join(infer_dir, "movie_embedder"))

    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history.history, f)

    print("=== Step 4.2B done (genre v2) ===")
    print("Saved inference models to:", infer_dir)

if __name__ == "__main__":
    main()
