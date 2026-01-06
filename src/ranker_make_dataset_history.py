import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss

# ---------------- Paths (HISTORY model) ----------------
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_history/inference/user_embedder"
FAISS_DIR = "../artifact/tfrs_retrival_model_history/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")

TRAIN_PATH = "../data/processed/train.parquet"
TEST_PATH  = "../data/processed/test.parquet"
OUT_PATH   = "../data/processed/ranker_dataset_history_hitonly.parquet"

# ---------------- Settings ----------------
CANDIDATE_K = 200
MAX_HIST = 50
MAX_USERS = 20000     # keep laptop-friendly
SEED = 42
BATCH_USERS = 256     # speed: batch user-model calls


def main():
    rng = np.random.default_rng(SEED)

    print("Loading history user model + FAISS...")
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    index = faiss.read_index(INDEX_PATH)

    with open(MOVIE_IDS_PATH, "r") as f:
        movie_ids = np.array(json.load(f), dtype=object)

    print("Loading train/test...")
    train_df = pd.read_parquet(TRAIN_PATH, columns=["userId", "movieId"])
    test_df  = pd.read_parquet(TEST_PATH,  columns=["userId", "movieId"])

    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)
    test_df["userId"] = test_df["userId"].astype(str)
    test_df["movieId"] = test_df["movieId"].astype(str)

    # user -> history list (from train)
    user_hist = train_df.groupby("userId")["movieId"].apply(list).to_dict()

    # popularity rank feature (from train)
    pop_counts = train_df["movieId"].value_counts()
    popular_rank = {mid: rank + 1 for rank, mid in enumerate(pop_counts.index.tolist())}

    # one test item per user
    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()
    if MAX_USERS is not None and len(test_pairs) > MAX_USERS:
        test_pairs = test_pairs.sample(MAX_USERS, random_state=SEED).reset_index(drop=True)

    print("Building HIT-only ranker dataset...")
    print("Users (sampled):", len(test_pairs), "Candidates/user:", CANDIDATE_K)

    rows = []
    group_id = 0
    hit_users = 0

    for start in range(0, len(test_pairs), BATCH_USERS):
        chunk = test_pairs.iloc[start:start + BATCH_USERS]

        users = chunk["userId"].tolist()
        true_movies = chunk["movieId"].tolist()

        # build histories
        hists = []
        valid_users = []
        valid_true = []

        for u, tm in zip(users, true_movies):
            hist = user_hist.get(u, [])
            if len(hist) == 0:
                continue
            valid_users.append(u)
            valid_true.append(tm)
            hists.append(hist[-MAX_HIST:])

        if not valid_users:
            continue

        hist_rt = tf.ragged.constant(hists, dtype=tf.string)
        u_vecs = user_model(hist_rt).numpy().astype("float32")  # (B, d)

        # retrieve candidates
        scores, idxs = index.search(u_vecs, CANDIDATE_K)         # (B, K)
        cand_movies = movie_ids[idxs]                            # (B, K)

        # for each user in batch, keep only HIT users
        for u, tm, cm_row, sc_row in zip(valid_users, valid_true, cand_movies, scores):
            # hit check
            if tm not in cm_row:
                continue

            hit_users += 1
            # make 200 training rows for this user
            for mid, sc in zip(cm_row, sc_row):
                rows.append({
                    "userId": u,
                    "movieId": mid,
                    "label": 1 if mid == tm else 0,
                    "retrieval_score": float(sc),
                    "pop_rank": int(popular_rank.get(mid, 10**9)),
                    "group": int(group_id),
                })
            group_id += 1

        if (start // BATCH_USERS + 1) % 20 == 0:
            done = min(start + BATCH_USERS, len(test_pairs))
            print(f"Processed {done}/{len(test_pairs)} users...  hit_users={hit_users}")

    ranker_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    ranker_df.to_parquet(OUT_PATH, index=False)

    print("\nSaved HIT-only ranker dataset to:", OUT_PATH)
    print("Shape:", ranker_df.shape)
    print("Groups(hit users):", group_id)
    print("Positives:", int(ranker_df["label"].sum()))
    if group_id > 0:
        print("Avg candidates/group:", ranker_df.shape[0] / group_id)


if __name__ == "__main__":
    main()
