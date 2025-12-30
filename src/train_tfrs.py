import os
import json
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd

TRAIN_PATH = "../data/processed/train.parquet"
VAL_PATH = "../data/processed/val.parquet"
MOVIES_PATH = "../data/processed/movies_clean.parquet"

OUT_DIR = "../artifact/tfrs_retrival_model"
os.makedirs(OUT_DIR, exist_ok = True)

BATCH_SIZE = 4096
EMBED_DIM = 64
EPOCHS = 10

def df_to_tfds(path: str) -> tf.data.Dataset:
    # read parquet via tensorflow
    df = pd.read_parquet(path, columns = ["userId", "movieId"])
    # convert strings, tfrs lookup layers like strings for stable vocab handling 
    
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)
    
    ds = tf.data.Dataset.from_tensor_slices({"userId": df["userId"].values, "movieId": df["movieId"].values})
    return ds

def build_vocab(ds: tf.data.Dataset, key: str):
    """Build a StringLookup vocabulary from dataset column."""
    vocab_ds = ds.map(lambda x: x[key])
    lookup = tf.keras.layers.StringLookup(mask_token=None, output_mode="int")  # ← Add output_mode
    lookup.adapt(vocab_ds.batch(100_000))
    return lookup

class TwoTower(tfrs.models.Model):
    def __init__(self, user_lookup, movie_lookup, candidate_ds):
        super().__init__()
        
        # Store lookups separately (for string -> integer conversion)
        self.user_lookup = user_lookup
        self.movie_lookup = movie_lookup
        
        # User tower (embedding + dense layers, WITHOUT lookup)
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(user_lookup.vocabulary_size(), EMBED_DIM),
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBED_DIM)
        ])

        # Movie tower (embedding + dense layers, WITHOUT lookup)
        self.movie_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(movie_lookup.vocabulary_size(), EMBED_DIM),
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBED_DIM)
        ])
        
        self.task = tfrs.tasks.Retrieval()
    
    # Properties for compatibility
    @property
    def user_model(self):
        return self.user_tower
    
    @property  
    def movie_model(self):
        return self.movie_tower
        
    def compute_loss(self, features, training=False):
        # Apply lookup FIRST (string -> integer)
        user_ids = self.user_lookup(features["userId"])
        movie_ids = self.movie_lookup(features["movieId"])
        
        # Then apply embedding towers
        user_emb = self.user_tower(user_ids)
        movie_emb = self.movie_tower(movie_ids)
        
        return self.task(user_emb, movie_emb)
        
def main():
    train_ds_raw = df_to_tfds(TRAIN_PATH)
    val_ds_raw = df_to_tfds(VAL_PATH)
    
    # Candidate dataset = unique movieIds from movies_clean
    movies_df = pd.read_parquet(MOVIES_PATH, columns = ["movieId"])
    movies_df["movieId"] = movies_df["movieId"].astype(str)
    candidate_ds = tf.data.Dataset.from_tensor_slices(movies_df["movieId"].values)
    
    # Build vocabularies from training data and movie list
    user_lookup = build_vocab(train_ds_raw, "userId")
    movie_lookup = tf.keras.layers.StringLookup(mask_token=None, output_mode="int")
    movie_lookup.adapt(candidate_ds.batch(100_000))
    
    # Prepare tarining pipelines
    train_ds = (
        train_ds_raw
        .shuffle(200_000, seed = 42, reshuffle_each_iteration = True)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    
    val_ds = (
        val_ds_raw
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    
    model = TwoTower(user_lookup, movie_lookup, candidate_ds)
    
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    
    # Save model
    model_path = os.path.join(OUT_DIR, "saved_model")
    tf.saved_model.save(model, model_path)
    
    # save training history
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history.history, f)
        
        
    print("=== Step 4 done: training and saving model ===")
    print("Model saved to:", model_path)
    print("Training history saved to:", os.path.join(OUT_DIR, "history.json"))

    # Create inference models
    infer_dir = os.path.join(OUT_DIR, "inference")
    os.makedirs(infer_dir, exist_ok = True)

    user_in = tf.keras.Input(shape=(), dtype=tf.string, name = "userId")
    movie_in = tf.keras.Input(shape=(), dtype=tf.string, name = "movieId")

    user_vec = model.user_tower(model.user_lookup(user_in))
    movie_vec = model.movie_tower(model.movie_lookup(movie_in))

    user_vec = tf.math.l2_normalize(user_vec, axis = 1)
    movie_vec = tf.math.l2_normalize(movie_vec, axis = 1)

    user_model = tf.keras.Model(user_in, user_vec, name="user_embedder")
    movie_model = tf.keras.Model(movie_in, movie_vec, name= "movie_embedder")

    tf.saved_model.save(user_model, os.path.join(infer_dir, "user_embedder"))
    tf.saved_model.save(movie_model, os.path.join(infer_dir, "movie_embedder"))


if __name__ == "__main__":
    main()