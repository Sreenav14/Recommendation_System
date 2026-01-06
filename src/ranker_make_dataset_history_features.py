import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss

# ---------------- Paths ----------------
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_history/inference/user_embedder"

FAISS_DIR      = "../artifact/tfrs_retrival_model_history/faiss"
INDEX_PATH     = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")

TRAIN_PATH      = "../data/processed/train.parquet"
TEST_PATH       = "../data/processed/test.parquet"
TRAIN_HIST_PATH = "../data/processed/train_hist.parquet"

OUT_PATH = "../data/processed/ranker_dataset_history_features.parquet"

# --------------- Settings ---------------
CANDIDATE_K = 200
MAX_USERS   = 20000
MAX_HIST    = 50
SEED        = 42


def normalize_user_id(x) -> str:
    # Make userId stable as string (handles int/float/string)
    # Example: 1.0 -> "1"
    try:
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
    except Exception:
        pass
    return str(x)


def to_str_list(x):
    """
    Convert hist_movieIds from parquet into a clean python list[str].
    Handles:
      - list
      - numpy.ndarray
      - None
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i) for i in list(x)]
    # last fallback: maybe it is already a string like "['1','2']"
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                # try json style first
                return [str(i) for i in json.loads(s)]
            except Exception:
                pass
    return []


def to_ragged_batch(list_of_lists):
    return tf.ragged.constant(list_of_lists, dtype=tf.string)


def main():
    rng = np.random.default_rng(SEED)

    print("Loading history user model + FAISS...")
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    index = faiss.read_index(INDEX_PATH)

    with open(MOVIE_IDS_PATH, "r") as f:
        movie_ids = np.array(json.load(f), dtype=object)

    # ---------------- Load data ----------------
    print("Loading train/test/history...")
    train_df = pd.read_parquet(TRAIN_PATH, columns=["userId", "movieId"])
    test_df  = pd.read_parquet(TEST_PATH, columns=["userId", "movieId"])
    hist_df  = pd.read_parquet(TRAIN_HIST_PATH, columns=["userId", "movieId", "hist_movieIds"])

    # normalize ids
    train_df["userId"] = train_df["userId"].apply(normalize_user_id)
    train_df["movieId"] = train_df["movieId"].astype(str)

    test_df["userId"] = test_df["userId"].apply(normalize_user_id)
    test_df["movieId"] = test_df["movieId"].astype(str)

    hist_df["userId"] = hist_df["userId"].apply(normalize_user_id)
    hist_df["movieId"] = hist_df["movieId"].astype(str)
    hist_df["hist_movieIds"] = hist_df["hist_movieIds"].apply(to_str_list)

    # popularity rank (from train)
    pop_counts = train_df["movieId"].value_counts()
    popular_rank = {mid: r + 1 for r, mid in enumerate(pop_counts.index.tolist())}

    # one test item per user
    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()

    # sample users
    if MAX_USERS is not None and len(test_pairs) > MAX_USERS:
        test_pairs = test_pairs.sample(MAX_USERS, random_state=SEED).reset_index(drop=True)

    # each user’s latest history list from train
    user_latest_hist = hist_df.groupby("userId")["hist_movieIds"].last().to_dict()

    # quick debug: check how many users have history
    hit_hist = sum(1 for u in test_pairs["userId"].tolist() if len(user_latest_hist.get(u, [])) > 0)
    print(f"Users in test_pairs: {len(test_pairs)}")
    print(f"Users with NON-empty history found: {hit_hist}")

    if hit_hist == 0:
        print("\nDEBUG: Example userIds from test_pairs:", test_pairs["userId"].head(5).tolist())
        print("DEBUG: Example userIds from hist_df:", list(user_latest_hist.keys())[:5])
        print("=> This means userId formats don't match OR hist_movieIds parsing is failing.")
        # still continue, but will likely produce empty output

    # map movieId -> FAISS row index
    movie_to_idx = {m: j for j, m in enumerate(movie_ids.tolist())}

    print("\nBuilding ranker dataset with history similarity features...")
    print("Users:", len(test_pairs), "Candidates/user:", CANDIDATE_K, "MAX_HIST:", MAX_HIST)

    rows = []

    for i, row in test_pairs.iterrows():
        user = row["userId"]
        true_movie = row["movieId"]

        hist_list = user_latest_hist.get(user, [])
        hist_list = hist_list[-MAX_HIST:]

        if len(hist_list) == 0:
            continue

        # user embedding from history
        hist_batch = to_ragged_batch([hist_list])
        u = user_model(hist_batch).numpy().astype("float32")  # (1, d)

        # retrieve candidates
        scores, idxs = index.search(u, CANDIDATE_K)
        idxs = idxs[0]
        scores = scores[0]
        cand_movies = movie_ids[idxs].copy()

        # ensure true movie is present
        if true_movie not in cand_movies:
            cand_movies[-1] = true_movie
            scores[-1] = float(scores.min() - 1e-3)

        # similarity features using FAISS reconstruct
        hist_idxs = [movie_to_idx.get(m) for m in hist_list if m in movie_to_idx]
        hist_idxs = [h for h in hist_idxs if h is not None]

        hist_mean = np.zeros(len(cand_movies), dtype=np.float32)
        hist_max  = np.zeros(len(cand_movies), dtype=np.float32)
        hist_last = np.zeros(len(cand_movies), dtype=np.float32)

        if len(hist_idxs) > 0:
            hist_vecs = np.vstack([index.reconstruct(int(h)) for h in hist_idxs]).astype("float32")
            cand_vecs = np.vstack([index.reconstruct(int(movie_to_idx[m])) for m in cand_movies]).astype("float32")
            sim = cand_vecs @ hist_vecs.T
            hist_mean = sim.mean(axis=1).astype(np.float32)
            hist_max  = sim.max(axis=1).astype(np.float32)
            hist_last = sim[:, -1].astype(np.float32)

        pop_vals = np.array([popular_rank.get(mid, 10**9) for mid in cand_movies])

        for mid, sc, pr, hm, hx, hl in zip(cand_movies, scores, pop_vals, hist_mean, hist_max, hist_last):
            rows.append({
                "userId": user,
                "movieId": mid,
                "label": 1 if mid == true_movie else 0,
                "retrieval_score": float(sc),
                "pop_rank": int(pr),
                "hist_sim_mean": float(hm),
                "hist_sim_max": float(hx),
                "hist_sim_last": float(hl),
                "group": int(i),
            })

        if (i + 1) % 2000 == 0:
            print(f"Processed {i+1}/{len(test_pairs)} users")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print("Shape:", df.shape)

    if "label" in df.columns:
        print("Positives:", int(df["label"].sum()))
        print("Groups:", df["group"].nunique())
    else:
        print("No rows were generated (still empty). Check the DEBUG output above.")


if __name__ == "__main__":
    main()
