import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import faiss

# ---- HISTORY paths ----
USER_MODEL_DIR = "../artifact/tfrs_retrival_model_history/inference/user_embedder"
FAISS_DIR = "../artifact/tfrs_retrival_model_history/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MOVIE_IDS_PATH = os.path.join(FAISS_DIR, "movie_ids.json")

TRAIN_PATH = "../data/processed/train.parquet"
TEST_PATH  = "../data/processed/test.parquet"

RANKER_MODEL_PATH = "../artifact/ranker_lgbm_history_hitonly/lgbm_ranker.txt"
OUT_PATH = "../artifact/ranker_lgbm_history_hitonly/eval_metrics.json"

# settings
CANDIDATE_K = 200
K_LIST = [10, 50]
MAX_USERS = 20000
MAX_HIST = 50
SEED = 42
BATCH_USERS = 256

def recall_at_k(ranks, k):
    return float(np.mean(ranks <= k))

def ndcg_at_k(ranks, k):
    vals = []
    for r in ranks:
        vals.append(1.0 / np.log2(r + 1) if r <= k else 0.0)
    return float(np.mean(vals))

def main():
    print("Loading user model, FAISS, and ranker...")
    user_model = tf.saved_model.load(USER_MODEL_DIR)
    index = faiss.read_index(INDEX_PATH)
    ranker = lgb.Booster(model_file=RANKER_MODEL_PATH)

    with open(MOVIE_IDS_PATH, "r") as f:
        movie_ids = np.array(json.load(f), dtype=object)

    print("Loading train/test...")
    train_df = pd.read_parquet(TRAIN_PATH, columns=["userId", "movieId"])
    test_df  = pd.read_parquet(TEST_PATH,  columns=["userId", "movieId"])

    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)
    test_df["userId"] = test_df["userId"].astype(str)
    test_df["movieId"] = test_df["movieId"].astype(str)

    # history from train
    user_hist = train_df.groupby("userId")["movieId"].apply(list).to_dict()

    # popularity rank
    pop_counts = train_df["movieId"].value_counts()
    popular_rank = {mid: rank + 1 for rank, mid in enumerate(pop_counts.index.tolist())}

    test_pairs = test_df.groupby("userId")["movieId"].first().reset_index()
    if MAX_USERS is not None and len(test_pairs) > MAX_USERS:
        test_pairs = test_pairs.sample(MAX_USERS, random_state=SEED).reset_index(drop=True)

    ranks_pop = []
    ranks_retr = []
    ranks_ranker = []
    hit_users = 0

    print("Evaluating HIT users only...")
    for start in range(0, len(test_pairs), BATCH_USERS):
        chunk = test_pairs.iloc[start:start + BATCH_USERS]

        users = chunk["userId"].tolist()
        true_movies = chunk["movieId"].tolist()

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

        u_vecs = user_model(tf.ragged.constant(hists, dtype=tf.string)).numpy().astype("float32")
        scores, idxs = index.search(u_vecs, CANDIDATE_K)
        cand_movies = movie_ids[idxs]  # (B, K)

        for u, tm, cm_row, sc_row in zip(valid_users, valid_true, cand_movies, scores):
            if tm not in cm_row:
                continue  # skip miss users

            hit_users += 1

            cm = cm_row
            sc = sc_row
            pop_vals = np.array([popular_rank.get(mid, 10**9) for mid in cm], dtype=np.int64)

            true_pos = int(np.where(cm == tm)[0][0])

            # popularity order (lower pop_rank is better)
            pop_order = np.argsort(pop_vals)
            pop_rank_pos = 1 + int(np.where(pop_order == true_pos)[0][0])
            ranks_pop.append(pop_rank_pos)

            # retrieval order (higher score is better)
            retr_order = np.argsort(-sc)
            retr_rank_pos = 1 + int(np.where(retr_order == true_pos)[0][0])
            ranks_retr.append(retr_rank_pos)

            # ranker
            X = np.stack([sc, pop_vals], axis=1)
            ranker_scores = ranker.predict(X)
            ranker_order = np.argsort(-ranker_scores)
            ranker_rank_pos = 1 + int(np.where(ranker_order == true_pos)[0][0])
            ranks_ranker.append(ranker_rank_pos)

        done = min(start + BATCH_USERS, len(test_pairs))
        if done % 5000 == 0:
            print(f"Processed {done}/{len(test_pairs)} users... hit_users={hit_users}")

    ranks_pop = np.array(ranks_pop)
    ranks_retr = np.array(ranks_retr)
    ranks_ranker = np.array(ranks_ranker)

    print("\n=== History HIT-only Pipeline Evaluation ===")
    print("Total users:", len(test_pairs))
    print("Hit users:", hit_users)
    print("Hit-rate@200:", hit_users / len(test_pairs))

    results = {
        "users_total": int(len(test_pairs)),
        "hit_users": int(hit_users),
        "hit_rate_at_200": float(hit_users / len(test_pairs)),
        "metrics": {}
    }

    for k in K_LIST:
        results["metrics"][str(k)] = {}
        for name, ranks in [
            ("popularity", ranks_pop),
            ("retrieval", ranks_retr),
            ("ranker", ranks_ranker),
        ]:
            results["metrics"][str(k)][name] = {
                "recall": recall_at_k(ranks, k),
                "ndcg": ndcg_at_k(ranks, k),
            }

        print(f"\nK={k} (HIT users only)")
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
