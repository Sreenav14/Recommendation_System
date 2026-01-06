import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss

# ---- Paths ----
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_history/inference/user_embedder"
FAISS_DIR = "../artifact/tfrs_retrival_model_history/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")

TEST_PATH = "../data/processed/val_hist.parquet"   # next item lives here
OUT_PATH = "../artifact/tfrs_retrival_model_history/eval_history_retrieval.json"

# ---- Settings ----
CANDIDATE_K = 200
MAX_HIST = 50
MAX_USERS = 20000
SEED = 42


def pad_or_trim_hist(hist, max_len=50):
    """Keep last max_len items, pad with '0' if needed."""
    hist = [str(x) for x in hist]

    # keep only last 50
    if len(hist) > max_len:
        hist = hist[-max_len:]

    # left-pad with "0" if shorter
    if len(hist) < max_len:
        hist = ["0"] * (max_len - len(hist)) + hist

    return hist


def main():
    print("Loading history user model + FAISS...")
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    index = faiss.read_index(INDEX_PATH)

    with open(MOVIE_IDS_PATH, "r") as f:
        movie_ids = np.array(json.load(f), dtype=object)

    print("Loading val_hist...")
    df = pd.read_parquet(TEST_PATH, columns=["userId", "movieId", "hist_movieIds"])
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)

    # keep only rows with non-empty history
    df = df[df["hist_movieIds"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)].reset_index(drop=True)

    if MAX_USERS is not None and len(df) > MAX_USERS:
        df = df.sample(MAX_USERS, random_state=SEED).reset_index(drop=True)

    print(f"Evaluating users: {len(df)}  |  K={CANDIDATE_K}  |  MAX_HIST={MAX_HIST}")

    hits = 0
    total = 0

    for i, row in df.iterrows():
        true_movie = row["movieId"]
        hist = row["hist_movieIds"]

        if isinstance(hist, np.ndarray):
            hist = hist.tolist()

        # make EXACT length 50
        hist = pad_or_trim_hist(hist, MAX_HIST)

        # IMPORTANT FIX:
        # user_model expects a dense tensor: (batch, 50)
        hist_tensor = tf.constant([hist], dtype=tf.string)   # shape = (1, 50)

        # user embedding
        u = user_model(hist_tensor).numpy().astype("float32")  # (1, d)

        # retrieve candidates
        scores, idxs = index.search(u, CANDIDATE_K)
        cand_movies = movie_ids[idxs[0]]

        if true_movie in cand_movies:
            hits += 1
        total += 1

        if (i + 1) % 5120 == 0:
            print(f"Processed {i+1}/{len(df)} users...")

    hit_rate = hits / max(total, 1)

    print("\n=== History Retrieval Evaluation ===")
    print("Users evaluated (with non-empty history):", total)
    print(f"Hit-rate@{CANDIDATE_K}: {hit_rate:.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "users_evaluated": int(total),
                "k": int(CANDIDATE_K),
                "max_hist": int(MAX_HIST),
                "hit_rate": float(hit_rate),
            },
            f,
            indent=2,
        )

    print("Saved metrics to:", OUT_PATH)


if __name__ == "__main__":
    main()
