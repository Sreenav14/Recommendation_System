import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# ---- Paths ----
USER_MODEL_DIR  = "../artifact/tfrs_retrival_model_genre/inference/user_embedder"
MOVIE_MODEL_DIR = "../artifact/tfrs_retrival_model_genre/inference/movie_embedder"

TRAIN_PATH = "../data/processed/train.parquet"
TEST_PATH  = "../data/processed/test.parquet"
MOVIES_PATH = "../data/processed/movies_clean.parquet"

OUT_METRICS_PATH = "../artifact/tfrs_retrival_model_genre/eval_metrics.json"

# ---- Settings ----
NEGATIVES_PER_USER = 500
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
    # Load inference models
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    movie_model = tf.saved_model.load(MOVIE_MODEL_DIR)

    # Load data
    train_df = pd.read_parquet(TRAIN_PATH, columns=["userId", "movieId"])
    test_df  = pd.read_parquet(TEST_PATH,  columns=["userId", "movieId"])
    movies_df = pd.read_parquet(MOVIES_PATH, columns=["movieId", "genres"])

    # Cast to strings
    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)
    test_df["userId"] = test_df["userId"].astype(str)
    test_df["movieId"] = test_df["movieId"].astype(str)

    movies_df["movieId"] = movies_df["movieId"].astype(str)
    movies_df["genres"] = movies_df["genres"].fillna("").astype(str)

    all_movies = movies_df["movieId"].unique()

    # MovieId -> genres string mapping
    movieid_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"]))

    # user -> seen movies set
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    # popularity baseline
    pop_counts = train_df["movieId"].value_counts()
    popular_rank = {mid: rank + 1 for rank, mid in enumerate(pop_counts.index.tolist())}

    # one test item per user
    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()

    rng = np.random.default_rng(42)
    ranks_model = []
    ranks_pop = []

    for _, row in test_pairs.iterrows():
        user = row["userId"]
        true_movie = row["movieId"]
        seen = user_seen.get(user, set())

        # sample negatives
        negatives = []
        while len(negatives) < NEGATIVES_PER_USER:
            cand = rng.choice(all_movies)
            if cand != true_movie and cand not in seen:
                negatives.append(cand)

        candidates = np.array([true_movie] + negatives, dtype=object)

        # Prepare genres for candidates
        cand_genres = np.array([movieid_to_genres.get(mid, "") for mid in candidates], dtype=object)

        # user embedding
        u = user_model(tf.constant([user], dtype=tf.string)).numpy()  # (1, d)

        # movie embeddings (need movieId + genres)
        V = movie_model([
            tf.constant(candidates, dtype=tf.string),
            tf.constant(cand_genres, dtype=tf.string),
        ]).numpy()  # (C, d)

        scores = (u @ V.T).reshape(-1)

        true_score = scores[0]
        rank = 1 + int(np.sum(scores > true_score))
        ranks_model.append(rank)

        # popularity rank inside same candidate set
        cand_pop = np.array([popular_rank.get(mid, 10**9) for mid in candidates])
        pop_rank = 1 + int(np.sum(cand_pop < cand_pop[0]))
        ranks_pop.append(pop_rank)

    ranks_model = np.array(ranks_model)
    ranks_pop = np.array(ranks_pop)

    print("=== Step 5: Offline evaluation (genre model, sampled) ===")
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
        print(f"Two-Tower(Genres) Recall@{k}: {r_model:.4f}   NDCG@{k}: {n_model:.4f}")
        print(f"Popularity        Recall@{k}: {r_pop:.4f}   NDCG@{k}: {n_pop:.4f}")

        results["metrics"][str(k)] = {
            "two_tower_genres": {"recall": float(r_model), "ndcg": float(n_model)},
            "popularity": {"recall": float(r_pop), "ndcg": float(n_pop)},
        }

    os.makedirs(os.path.dirname(OUT_METRICS_PATH), exist_ok=True)
    with open(OUT_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved metrics to:", OUT_METRICS_PATH)

if __name__ == "__main__":
    main()
