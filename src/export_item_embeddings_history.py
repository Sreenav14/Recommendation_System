import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

MOVIE_MODEL_DIR = "../artifact/tfrs_retrival_model_history/inference/movie_embedder"
MOVIES_PATH = "../data/processed/movies_clean.parquet"

OUT_DIR = "../artifact/tfrs_retrival_model_history/faiss"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    movie_model = tf.saved_model.load(MOVIE_MODEL_DIR)

    movies_df = pd.read_parquet(MOVIES_PATH, columns=["movieId"])
    movies_df["movieId"] = movies_df["movieId"].astype(str)
    movie_ids = movies_df["movieId"].values

    batch_size = 8192
    all_vecs = []

    for i in range(0, len(movie_ids), batch_size):
        batch_ids = movie_ids[i:i + batch_size]
        vecs = movie_model(tf.constant(batch_ids, dtype=tf.string)).numpy()
        all_vecs.append(vecs)

    item_vecs = np.vstack(all_vecs).astype("float32")

    np.save(os.path.join(OUT_DIR, "items_embeddings.npy"), item_vecs)
    with open(os.path.join(OUT_DIR, "movie_ids.json"), "w") as f:
        json.dump(movie_ids.tolist(), f)

    print("saved embeddings to", os.path.join(OUT_DIR, "items_embeddings.npy"))
    print("saved movie_ids to", os.path.join(OUT_DIR, "movie_ids.json"))
    print("embeddings shape", item_vecs.shape)

if __name__ == "__main__":
    main()
