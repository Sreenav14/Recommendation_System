import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import faiss

# -----------------------------
# Paths
# -----------------------------
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_hardneg/inference/user_embedder"

FAISS_DIR = "../artifact/tfrs_retrival_model_hardneg/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")

TRAIN_INTERACTIONS_PATH = "../data/processed/train.parquet"
TEST_PATH = "../data/processed/test.parquet"

RANKER_MODEL_PATH = "../artifact/ranker_lgbm/lgbm_ranker.txt"
OUT_PATH = "../artifact/ranker_lgbm/eval_ranker_metrics_clean.json"

# -----------------------------
# Settings
# -----------------------------
CANDIDATES_K = 200
K_LIST = [10, 50]
MAX_USERS = 20000
SEED = 42

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
    print("Loading user model, FAISS, and ranker...")
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    index = faiss.read_index(INDEX_PATH)

    with open(MOVIE_IDS_PATH, "r") as f:
        movie_ids = np.array(json.load(f), dtype=object)

    ranker = lgb.Booster(model_file=RANKER_MODEL_PATH)

    print("Loading train/test data...")
    train_df = pd.read_parquet(TRAIN_INTERACTIONS_PATH, columns=["userId", "movieId"])
    test_df  = pd.read_parquet(TEST_PATH, columns=["userId", "movieId"])

    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)
    test_df["userId"]  = test_df["userId"].astype(str)
    test_df["movieId"] = test_df["movieId"].astype(str)

    # Popularity baseline from REAL interactions
    pop_counts = train_df["movieId"].value_counts()
    popular_rank = {mid: rank + 1 for rank, mid in enumerate(pop_counts.index.tolist())}

    # One test item per user
    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()

    if MAX_USERS is not None and len(test_pairs) > MAX_USERS:
        test_pairs = test_pairs.sample(n=MAX_USERS, random_state=SEED).reset_index(drop=True)

    print("Users evaluated:", len(test_pairs))
    print("Candidates/user:", CANDIDATES_K)

    # We will store ranks ONLY for users where retrieval hit happens
    ranks_pop_hit = []
    ranks_retrieval_hit = []
    ranks_ranker_hit = []

    retrieval_hits = 0

    for i, row in test_pairs.iterrows():
        user = row["userId"]
        true_movie = row["movieId"]

        # 1) User embedding
        u = user_model(tf.constant([user], dtype=tf.string)).numpy().astype("float32")

        # 2) Retrieve candidates
        scores, idxs = index.search(u, CANDIDATES_K)
        idxs = idxs[0]
        scores = scores[0]
        cand_movies = movie_ids[idxs]

        # If the true movie is NOT in retrieved candidates, ranker cannot help
        # so we count miss and skip ranking metrics for this user
        hit_positions = np.where(cand_movies == true_movie)[0]
        if len(hit_positions) == 0:
            continue

        retrieval_hits += 1
        true_idx = hit_positions[0]

        # Popularity inside candidate set
        pop_vals = np.array([popular_rank.get(mid, 10**9) for mid in cand_movies])
        pop_order = np.argsort(pop_vals)
        pop_rank_pos = 1 + int(np.where(pop_order == true_idx)[0][0])
        ranks_pop_hit.append(pop_rank_pos)

        # Retrieval-only rank (higher score better)
        retr_order = np.argsort(-scores)
        retr_rank_pos = 1 + int(np.where(retr_order == true_idx)[0][0])
        ranks_retrieval_hit.append(retr_rank_pos)

        # Ranker rank
        X = np.stack([scores, pop_vals], axis=1)
        ranker_scores = ranker.predict(X)
        ranker_order = np.argsort(-ranker_scores)
        ranker_rank_pos = 1 + int(np.where(ranker_order == true_idx)[0][0])
        ranks_ranker_hit.append(ranker_rank_pos)

        if (i + 1) % 2000 == 0:
            print(f"Processed {i+1}/{len(test_pairs)} users")

    total_users = len(test_pairs)
    hit_rate = retrieval_hits / total_users if total_users > 0 else 0.0

    ranks_pop_hit = np.array(ranks_pop_hit)
    ranks_retrieval_hit = np.array(ranks_retrieval_hit)
    ranks_ranker_hit = np.array(ranks_ranker_hit)

    print("\n=== Step 7C.2: Clean pipeline evaluation (ONLY retrieval hits) ===")
    print(f"Total users: {total_users}")
    print(f"Retrieval hits: {retrieval_hits}")
    print(f"Retriever hit-rate@{CANDIDATES_K}: {hit_rate:.4f}")

    results = {
        "users_total": int(total_users),
        "candidates_per_user": int(CANDIDATES_K),
        "retrieval_hits": int(retrieval_hits),
        "retriever_hit_rate": float(hit_rate),
        "metrics_on_hits": {}
    }

    if retrieval_hits == 0:
        print("No hits. Ranker evaluation not possible.")
    else:
        for k in K_LIST:
            results["metrics_on_hits"][str(k)] = {}

            for name, ranks in [
                ("popularity", ranks_pop_hit),
                ("retrieval_only", ranks_retrieval_hit),
                ("ranker", ranks_ranker_hit),
            ]:
                r = recall_at_k(ranks, k)
                n = ndcg_at_k(ranks, k)
                results["metrics_on_hits"][str(k)][name] = {"recall": float(r), "ndcg": float(n)}

            print(f"\nK={k} (on hit users only)")
            print(f"Popularity     Recall@{k}: {results['metrics_on_hits'][str(k)]['popularity']['recall']:.4f} "
                  f"NDCG@{k}: {results['metrics_on_hits'][str(k)]['popularity']['ndcg']:.4f}")
            print(f"Retrieval-only Recall@{k}: {results['metrics_on_hits'][str(k)]['retrieval_only']['recall']:.4f} "
                  f"NDCG@{k}: {results['metrics_on_hits'][str(k)]['retrieval_only']['ndcg']:.4f}")
            print(f"Ranker         Recall@{k}: {results['metrics_on_hits'][str(k)]['ranker']['recall']:.4f} "
                  f"NDCG@{k}: {results['metrics_on_hits'][str(k)]['ranker']['ndcg']:.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved clean ranker evaluation metrics to:", OUT_PATH)

if __name__ == "__main__":
    main()
