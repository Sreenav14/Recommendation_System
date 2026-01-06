import os
import numpy as np
import faiss
import json
import pandas as pd
import tensorflow as tf

# paths
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_hardneg/inference/user_embedder"
FAISS_DIR = "../artifact/tfrs_retrival_model_hardneg/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")
MOVIES_META_PATH = "../data/processed/movies_clean.parquet"

TOP_K = 200

def main():
    # load user model
    
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    
    # load faiss index
    index = faiss.read_index(INDEX_PATH)
    
    # load movie id mapping
    with open(MOVIE_IDS_PATH, "r") as f:
        movie_ids = json.load(f)
        
    # load movie metadata
    movies_pd = pd.read_parquet(MOVIES_META_PATH, columns = ["movieId", "title"])
    movies_pd["movieId"] = movies_pd["movieId"].astype(str)
    id_to_title = dict(zip(movies_pd["movieId"], movies_pd["title"]))
    
    # choose a random user
    user_id = input("Enter a user ID: ").strip()
    
    # compute user embedding
    u = user_model(tf.constant([user_id], dtype=tf.string)).numpy()
    
    # Search FAISS
    scores, indices = index.search(u, TOP_K)
    
    print(f"\n Top recommendations:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start = 1):
        mid = movie_ids[idx]
        title = id_to_title.get(mid, "Unknown Title")
        print(f"{rank:02d}. {title} (movieId = {mid} score = {score:.4f})")
        
        
if __name__ == "__main__":
    main()