import os
import json
import numpy as np 
import pandas as pd
import tensorflow as tf
import faiss

# paths
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_hardneg/inference/user_embedder"
FAISS_DIR = "../artifact/tfrs_retrival_model_hardneg/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")

TRAIN_PATH = "../data/processed/train.parquet"
TEST_PATH = "../data/processed/test.parquet"

OUT_PATH = "../data/processed/ranker_dataset.parquet"

CANDIDATE_K = 200
MAX_USERS = 20000
SEED = 42

def main():
    rng = np.random.default_rng(SEED)
    
    # load model + index
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    index = faiss.read_index(INDEX_PATH)
    
    # Load movieId mapping
    with open(MOVIE_IDS_PATH, "r") as f:
        movie_ids = np.array(json.load(f), dtype = object)
        
    # load interactions
    train_df = pd.read_parquet(TRAIN_PATH, columns = ["userId", "movieId"])
    test_df = pd.read_parquet(TEST_PATH, columns = ["userId", "movieId"])
    
    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)
    test_df["userId"] = test_df["userId"].astype(str)
    test_df["movieId"] = test_df["movieId"].astype(str)
    
    # user seen movies
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    
    # popularity rank feature
    pop_counts = train_df["movieId"].value_counts()
    popular_rank = {mid:rank +1 for rank, mid in enumerate(pop_counts.index.tolist())}
    
    # one test item per user
    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()
    
    # sample users for speed
    if MAX_USERS is not None and len(test_pairs) > MAX_USERS:
        test_pairs = test_pairs.sample(MAX_USERS, random_state = SEED).reset_index(drop = True)
        
    rows = []
    print("Building ranker dataset...")
    print("Users:", len(test_pairs), "Candidates/user:", CANDIDATE_K)
    
    for i, row in test_pairs.iterrows():
        user = row["userId"]
        true_movie = row["movieId"]
        
        #  1. Get User Embedding
        u = user_model(tf.constant([user], dtype = tf.string)).numpy().astype("float32")
        
        # 2. Retrieve top-k candidates from FAISS
        scores, idxs = index.search(u, CANDIDATE_K)
        idx = idxs[0]
        scores = scores[0]
        
        cand_movies = movie_ids[idx]
        
        if true_movie not in cand_movies:
            cand_movies[-1] = true_movie
            scores[-1] = float(scores.min() - 1e-3)
        
        # build labeled rows
        #  lable = 1 if candidate is the true movie
        for mid, sc in zip(cand_movies, scores):
            rows.append({
                "userId" : user,
                "movieId" : mid,
                "label" : 1 if mid == true_movie else 0,
                "retrieval_score" : float(sc),
                "pop_rank" : popular_rank.get(mid, 10**9),
                "group" : int(i),
            })
        
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1}/{len(test_pairs)} users")
        
    ranker_df = pd.DataFrame(rows)
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok = True)
    ranker_df.to_parquet(OUT_PATH, index = False)
    
    print("\n Saved ranker training dataset to: ", OUT_PATH)
    print("Shape:", ranker_df.shape)
    print("Postives:", ranker_df["label"].sum())
    print("Avg Candidates/user:", ranker_df.shape[0] / len(test_pairs))

if __name__ == "__main__":
    main()
        