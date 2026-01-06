import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss

# ---------------- Paths ----------------
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_history/inference/user_embedder"
FAISS_DIR = "../artifact/tfrs_retrival_model_history/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")

TRAIN_PATH = "../data/processed/train.parquet"
TEST_PATH  = "../data/processed/test.parquet"

# ---------------- Settings ----------------
K = 200                 # candidate set size for retrieval
MAX_HIST = 50           # must match your training cap, or pick 50 as standard
MAX_USERS = 20000       # keep it fast on laptop
SEED = 42


def make_ragged_hist(hist_list):
    """
    hist_list = list of lists of strings
    returns RaggedTensor shape [batch, None]
    """
    return tf.ragged.constant(hist_list, dtype=tf.string)


def main():
    rng = np.random.default_rng(SEED)

    print("Loading user model + FAISS...")
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

    # Build user -> history list from TRAIN (ordered doesn't matter for v1; we just need "what they saw")
    user_hist = train_df.groupby("userId")["movieId"].apply(list).to_dict()

    # One test movie per user (the “future” movie we hope retrieval can find)
    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()

    if MAX_USERS is not None and len(test_pairs) > MAX_USERS:
        test_pairs = test_pairs.sample(MAX_USERS, random_state=SEED).reset_index(drop=True)

    hits = 0
    evaluated = 0

    print(f"Evaluating users: {len(test_pairs)}  |  K={K}  |  MAX_HIST={MAX_HIST}")

    # Batch users for speed (calling TF per user is slow)
    BATCH_USERS = 256

    for start in range(0, len(test_pairs), BATCH_USERS):
        chunk = test_pairs.iloc[start:start + BATCH_USERS]

        users = chunk["userId"].tolist()
        true_movies = chunk["movieId"].tolist()

        # Build histories for this chunk
        hists = []
        keep_mask = []  # users who actually have history

        for u in users:
            hist = user_hist.get(u, [])
            if len(hist) == 0:
                keep_mask.append(False)
                hists.append(["__EMPTY__"])  # placeholder, will be dropped
            else:
                keep_mask.append(True)
                hists.append(hist[-MAX_HIST:])

        # Filter out empty-history users
        valid_idx = [i for i, ok in enumerate(keep_mask) if ok]
        if not valid_idx:
            continue

        true_movies_valid = [true_movies[i] for i in valid_idx]
        hists_valid = [hists[i] for i in valid_idx]

        # User embeddings from history (ragged)
        hist_rt = make_ragged_hist(hists_valid)
        u_vecs = user_model(hist_rt).numpy().astype("float32")   # (B, d)

        # FAISS search
        scores, idxs = index.search(u_vecs, K)                  # (B, K)
        cand_movies = movie_ids[idxs]                           # (B, K)

        # Check hits
        for cm, tm in zip(cand_movies, true_movies_valid):
            evaluated += 1
            if tm in cm:
                hits += 1

        if (start // BATCH_USERS + 1) % 20 == 0:
            print(f"Processed {min(start + BATCH_USERS, len(test_pairs))}/{len(test_pairs)} users...")

    hit_rate = hits / max(evaluated, 1)

    print("\n=== History Retrieval Evaluation ===")
    print("Users evaluated (with non-empty history):", evaluated)
    print(f"Hit-rate@{K}: {hit_rate:.4f}")

if __name__ == "__main__":
    main()
