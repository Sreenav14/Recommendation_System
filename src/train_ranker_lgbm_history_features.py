import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_PATH = "../data/processed/ranker_dataset_history_features.parquet"

OUT_DIR = "../artifact/ranker_lgbm_history_features"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "lgbm_ranker.txt")
FEATS_PATH = os.path.join(OUT_DIR, "features.json")

SEED = 42

FEATURES = [
    "retrieval_score",
    "pop_rank",
    "hist_sim_mean",
    "hist_sim_max",
    "hist_sim_last",
]

def main():
    print("Loading ranker dataset...")
    df = pd.read_parquet(DATA_PATH)

    print("ROWS:", len(df))
    print("GROUPS:", df["group"].nunique())
    print("Avg group size:", len(df) / df["group"].nunique())
    print("Positives:", int(df["label"].sum()))

    # Sort by group so LightGBM group structure is correct
    df = df.sort_values(["group", "label"], ascending=[True, False]).reset_index(drop=True)

    # Prepare X/y
    X = df[FEATURES].astype(np.float32)
    y = df["label"].astype(np.int32)

    # Build group sizes array
    group_sizes = df.groupby("group").size().values

    # Train/valid split by groups (not by rows)
    unique_groups = df["group"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_groups)

    n_valid = int(0.2 * len(unique_groups))
    valid_groups = set(unique_groups[:n_valid])
    train_groups = set(unique_groups[n_valid:])

    train_mask = df["group"].isin(train_groups)
    valid_mask = df["group"].isin(valid_groups)

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]

    train_group_sizes = df[train_mask].groupby("group").size().values
    valid_group_sizes = df[valid_mask].groupby("group").size().values

    print(f"Train queries: {len(train_group_sizes)} | Valid queries: {len(valid_group_sizes)}")

    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_group_sizes, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, group=valid_group_sizes, reference=lgb_train, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": ["ndcg"],
        "ndcg_eval_at": [10, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": SEED,
    }

    print("Training LightGBM ranker (history features)...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    model.save_model(MODEL_PATH)
    with open(FEATS_PATH, "w") as f:
        json.dump(FEATURES, f, indent=2)

    print("\nSaved ranker model to:", MODEL_PATH)
    print("Saved features list to:", FEATS_PATH)

if __name__ == "__main__":
    main()
