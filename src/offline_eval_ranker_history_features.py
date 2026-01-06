import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import faiss

# ---------------- Paths ----------------
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_history/inference/user_embedder"
MOVIE_MODEL_DIR = "../artifact/tfrs_retrival_model_history/inference/movie_embedder"

FAISS_DIR = "../artifact/tfrs_retrival_model_history/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")

TRAIN_PATH = "../data/processed/train.parquet"
TEST_PATH  = "../data/processed/test.parquet"

TRAIN_HIST_PATH = "../data/processed/train_hist.parquet"  # has hist_movieIds column

RANKER_MODEL_PATH = "../artifact/ranker_lgbm_history_features/lgbm_ranker.txt"
OUT_PATH = "../artifact/ranker_lgbm_history_features/eval_metrics.json"

# ---------------- Settings ----------------
CANDIDATE_K = 200
K_LIST = [10, 50]
MAX_USERS = 20000
MAX_HIST = 50
SEED = 42

FEATURES = ["retrieval_score", "pop_rank", "hist_sim_mean", "hist_sim_max", "hist_sim_last"]

def recall_at_k(ranks, k):
    return float(np.mean(ranks <= k))

def ndcg_at_k(ranks, k):
    vals = []
    for r in ranks:
        vals.append(1.0 / np.log2(r + 1) if r <= k else 0.0)
    return float(np.mean(vals))

def batched_movie_embeds(movie_model, movie_ids, batch_size=8192):
    out = []
    for i in range(0, len(movie_ids), batch_size):
        batch = movie_ids[i:i+batch_size]
        vecs = movie_model(tf.constant(batch, dtype=tf.string)).numpy().astype("float32")
        out.append(vecs)
    return np.vstack(out)

def main():
    rng = np.random.default_rng(SEED)

    print("Loading user/movie models, FAISS, and ranker...")
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    movie_model = tf.saved_model.load(MOVIE_MODEL_DIR)
    index = faiss.read_index(INDEX_PATH)

    with open(MOVIE_IDS_PATH, "r") as f:
        movie_ids = np.array(json.load(f), dtype=object)

    ranker = lgb.Booster(model_file=RANKER_MODEL_PATH)

    print("Loading train/test/history...")
    train_df = pd.read_parquet(TRAIN_PATH, columns=["userId", "movieId"])
    test_df  = pd.read_parquet(TEST_PATH,  columns=["userId", "movieId"])
    hist_df  = pd.read_parquet(TRAIN_HIST_PATH, columns=["userId", "movieId", "hist_movieIds"])

    # strings
    for df in [train_df, test_df, hist_df]:
        df["userId"] = df["userId"].astype(str)
        df["movieId"] = df["movieId"].astype(str)

    # user -> seen set (for sampling sanity if needed later)
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    # popularity rank (lower = more popular)
    pop_counts = train_df["movieId"].value_counts()
    popular_rank = {mid: rank + 1 for rank, mid in enumerate(pop_counts.index.tolist())}

    # user -> history list for the specific (userId, movieId) training row
    # We want user history for each test user. We'll take the last available history row for that user.
    # hist_df is built so that each row corresponds to an interaction with history.
    user_to_hist = hist_df.groupby("userId")["hist_movieIds"].last().to_dict()

    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()

    if MAX_USERS is not None and len(test_pairs) > MAX_USERS:
        test_pairs = test_pairs.sample(MAX_USERS, random_state=SEED).reset_index(drop=True)

    print(f"Users evaluated: {len(test_pairs)} | Candidates/user: {CANDIDATE_K} | MAX_HIST: {MAX_HIST}")

    ranks_pop = []
    ranks_retrieval = []
    ranks_ranker = []

    hit_users = 0

    for i, row in test_pairs.iterrows():
        user = row["userId"]
        true_movie = row["movieId"]

        hist_list = user_to_hist.get(user, [])
        if hist_list is None:
            hist_list = []
        # cap history
        hist_list = list(hist_list)[-MAX_HIST:]

        # if no history, skip (this is history-based pipeline)
        if len(hist_list) == 0:
            continue

        # 1) user embedding (history model may expect ragged [B, T])
        # We'll pass history list as a ragged batch of 1 user
        hist_rt = tf.ragged.constant([hist_list], dtype=tf.string)
        u = user_model(hist_rt).numpy().astype("float32")  # (1, d)

        # 2) retrieve candidates
        scores, idxs = index.search(u, CANDIDATE_K)
        idxs = idxs[0]
        scores = scores[0]
        cand_movies = movie_ids[idxs].copy()

        # ensure true movie is inside candidate set for evaluation stability
        if true_movie not in cand_movies:
            # swap last one
            cand_movies[-1] = true_movie
            scores[-1] = float(scores.min() - 1e-3)
        else:
            hit_users += 1

        # 3) compute history similarity features:
        # embed history movies once, then candidate movies, then dot products
        hist_emb = batched_movie_embeds(movie_model, np.array(hist_list, dtype=object), batch_size=2048)  # (H, d)
        cand_emb = batched_movie_embeds(movie_model, np.array(cand_movies, dtype=object), batch_size=8192)  # (C, d)

        # sims: (C, H)
        sims = cand_emb @ hist_emb.T
        hist_sim_mean = sims.mean(axis=1)
        hist_sim_max = sims.max(axis=1)
        hist_sim_last = sims[:, -1]  # similarity to most recent

        # 4) pop_rank feature
        pop_vals = np.array([popular_rank.get(mid, 10**9) for mid in cand_movies], dtype=np.float32)

        # 5) ranks inside candidate set
        true_idx = int(np.where(cand_movies == true_movie)[0][0])

        # popularity baseline (lower pop_rank is better)
        pop_order = np.argsort(pop_vals)
        pop_rank_pos = 1 + int(np.where(pop_order == true_idx)[0][0])
        ranks_pop.append(pop_rank_pos)

        # retrieval baseline (higher score better)
        retr_order = np.argsort(-scores)
        retr_rank_pos = 1 + int(np.where(retr_order == true_idx)[0][0])
        ranks_retrieval.append(retr_rank_pos)

        # ranker
        X = np.stack([scores, pop_vals, hist_sim_mean, hist_sim_max, hist_sim_last], axis=1).astype(np.float32)
        ranker_scores = ranker.predict(X)
        ranker_order = np.argsort(-ranker_scores)
        ranker_rank_pos = 1 + int(np.where(ranker_order == true_idx)[0][0])
        ranks_ranker.append(ranker_rank_pos)

        if (i + 1) % 2000 == 0:
            print(f"Processed {i+1}/{len(test_pairs)} users... hit_users={hit_users}")

    ranks_pop = np.array(ranks_pop)
    ranks_retrieval = np.array(ranks_retrieval)
    ranks_ranker = np.array(ranks_ranker)

    results = {
        "users_evaluated_total": int(len(test_pairs)),
        "users_used_nonempty_history": int(len(ranks_ranker)),
        "hit_users": int(hit_users),
        "hit_rate_at_200": float(hit_users / max(1, len(ranks_ranker))),
        "candidates_per_user": int(CANDIDATE_K),
        "max_hist": int(MAX_HIST),
        "metrics": {}
    }

    print("\n=== History + Ranker Pipeline Evaluation ===")
    print("Users used (non-empty history):", len(ranks_ranker))
    print("Hit users:", hit_users)
    print("Hit-rate@200:", results["hit_rate_at_200"])

    for k in K_LIST:
        results["metrics"][str(k)] = {}
        for name, ranks in [
            ("popularity", ranks_pop),
            ("retrieval", ranks_retrieval),
            ("ranker", ranks_ranker),
        ]:
            results["metrics"][str(k)][name] = {
                "recall": recall_at_k(ranks, k),
                "ndcg": ndcg_at_k(ranks, k),
            }

        print(f"\nK={k}")
        print(f"Popularity     Recall@{k}: {results['metrics'][str(k)]['popularity']['recall']:.4f} "
              f"NDCG@{k}: {results['metrics'][str(k)]['popularity']['ndcg']:.4f}")
        print(f"Retrieval-only Recall@{k}: {results['metrics'][str(k)]['retrieval']['recall']:.4f} "
              f"NDCG@{k}: {results['metrics'][str(k)]['retrieval']['ndcg']:.4f}")
        print(f"Ranker         Recall@{k}: {results['metrics'][str(k)]['ranker']['recall']:.4f} "
              f"NDCG@{k}: {results['metrics'][str(k)]['ranker']['ndcg']:.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved metrics to:", OUT_PATH)

if __name__ == "__main__":
    main()
