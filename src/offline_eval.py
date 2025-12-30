import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# ---- Paths (adjust if your folder name differs) ----
USER_MODEL_DIR  = "../artifact/tfrs_retrival_model_hardneg/inference/user_embedder"
MOVIE_MODEL_DIR = "../artifact/tfrs_retrival_model_hardneg/inference/movie_embedder"

TRAIN_PATH = "../data/processed/train.parquet"
TEST_PATH  = "../data/processed/test.parquet"
MOVIES_PATH = "../data/processed/movies_clean.parquet"

OUT_METRICS_PATH = "../artifact/tfrs_retrival_model_hardneg/eval_metrics.json"

# ---- Settings ----
NEGATIVES_PER_USER = 500  # balanced default
K_LIST = [10, 50]

def recall_at_k(ranks, k):
    return float(np.mean(ranks <= k))

def ndcg_at_k(ranks, k):
    vals = []
    for r in ranks:
        if r <= k:
            vals.append(1.0 / np.log2(r + 1))
        else:
            vals.append(0.0)
    return float(np.mean(vals))

def main():
    # -----------------------------
    # Load inference models (callable)
    # -----------------------------
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    movie_model = tf.saved_model.load(MOVIE_MODEL_DIR)

    # -----------------------------
    # Load data
    # -----------------------------
    train_df = pd.read_parquet(TRAIN_PATH, columns=["userId", "movieId"])
    test_df  = pd.read_parquet(TEST_PATH,  columns=["userId", "movieId"])
    movies_df = pd.read_parquet(MOVIES_PATH, columns=["movieId"])

    # Your inference models expect strings (because you used StringLookup)
    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)
    test_df["userId"] = test_df["userId"].astype(str)
    test_df["movieId"] = test_df["movieId"].astype(str)
    movies_df["movieId"] = movies_df["movieId"].astype(str)

    all_movies = movies_df["movieId"].unique()
    all_movies_set = set(all_movies.tolist())

    # user -> set of movies they already liked in train
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    # popularity baseline ranking
    pop_counts = train_df["movieId"].value_counts()
    top_popular_movies = pop_counts.index.tolist()[:5000]  # top 5k popular
    top_popular_movies = np.array(top_popular_movies, dtype=object)
    popular_rank = {mid: rank + 1 for rank, mid in enumerate(pop_counts.index.tolist())}

    # one test item per user
    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()

    rng = np.random.default_rng(42)
    ranks_model = []
    ranks_pop = []

    # -----------------------------
    # Evaluate user by user (simple + clear)
    # -----------------------------
    for _, row in test_pairs.iterrows():
        user = row["userId"]
        true_movie = row["movieId"]

        seen = user_seen.get(user, set())

        # sample negatives that user did not already see
        negatives = []
        while len(negatives) < NEGATIVES_PER_USER:
            cand = rng.choice(top_popular_movies)
            if cand != true_movie and cand not in seen:
                negatives.append(cand)

        candidates = np.array([true_movie] + negatives, dtype=object)

        # user embedding
        u = user_model(tf.constant([user], dtype=tf.string)).numpy()  # (1, d)

        # movie embeddings (all candidates)
        V = movie_model(tf.constant(candidates, dtype=tf.string)).numpy()  # (C, d)

        # scores = dot product
        scores = (u @ V.T).reshape(-1)

        # rank of true movie (index 0)
        true_score = scores[0]
        rank = 1 + int(np.sum(scores > true_score))
        ranks_model.append(rank)

        # popularity baseline rank within same candidate set
        cand_pop = np.array([popular_rank.get(mid, 10**9) for mid in candidates])
        pop_rank = 1 + int(np.sum(cand_pop < cand_pop[0]))
        ranks_pop.append(pop_rank)

    ranks_model = np.array(ranks_model)
    ranks_pop = np.array(ranks_pop)

    # -----------------------------
    # Print metrics
    # -----------------------------
    print("=== Step 5: Offline evaluation (sampled) ===")
    print(f"Users evaluated: {len(ranks_model):,}")
    print(f"Negatives/user: {NEGATIVES_PER_USER}")

    results = {
        "users_evaluated": int(len(ranks_model)),
        "negatives_per_user": int(NEGATIVES_PER_USER),
        "metrics": {}
    }

    for k in K_LIST:
        r_model = recall_at_k(ranks_model, k)
        n_model = ndcg_at_k(ranks_model, k)
        r_pop = recall_at_k(ranks_pop, k)
        n_pop = ndcg_at_k(ranks_pop, k)

        print(f"\nK={k}")
        print(f"Two-Tower   Recall@{k}: {r_model:.4f}   NDCG@{k}: {n_model:.4f}")
        print(f"Popularity  Recall@{k}: {r_pop:.4f}   NDCG@{k}: {n_pop:.4f}")

        results["metrics"][str(k)] = {
            "two_tower": {"recall": float(r_model), "ndcg": float(n_model)},
            "popularity": {"recall": float(r_pop), "ndcg": float(n_pop)},
        }

    os.makedirs(os.path.dirname(OUT_METRICS_PATH), exist_ok=True)
    with open(OUT_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved metrics to:", OUT_METRICS_PATH)

if __name__ == "__main__":
    main()
