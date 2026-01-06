import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_PATH = "../data/processed/ranker_dataset_history_hitonly.parquet"
OUT_DIR = "../artifact/ranker_lgbm_history_hitonly"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "lgbm_ranker.txt")
FEATURES_PATH = os.path.join(OUT_DIR, "features.json")

# We keep it simple and consistent with your earlier ranker:
FEATURE_COLS = ["retrieval_score", "pop_rank"]  # same as before
LABEL_COL = "label"
GROUP_COL = "group"

def main():
    print("Loading HIT-only ranker dataset...")
    df = pd.read_parquet(DATA_PATH)

    print("ROWS:", len(df))
    print("GROUPS:", df[GROUP_COL].nunique())
    print("Avg group size:", len(df) / df[GROUP_COL].nunique())
    print("Positives:", int(df[LABEL_COL].sum()))

    # Sort by group so LightGBM understands query boundaries correctly
    df = df.sort_values(GROUP_COL).reset_index(drop=True)

    # Train/valid split by groups (not rows) — very important for ranking
    groups = df[GROUP_COL].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(groups)

    n_valid = max(1, int(0.2 * len(groups)))
    valid_groups = set(groups[:n_valid])
    train_groups = set(groups[n_valid:])

    train_df = df[df[GROUP_COL].isin(train_groups)].copy()
    valid_df = df[df[GROUP_COL].isin(valid_groups)].copy()

    # Build group arrays: number of rows per query(group)
    train_group_sizes = train_df.groupby(GROUP_COL).size().to_numpy()
    valid_group_sizes = valid_df.groupby(GROUP_COL).size().to_numpy()

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[LABEL_COL].to_numpy()

    X_valid = valid_df[FEATURE_COLS].to_numpy()
    y_valid = valid_df[LABEL_COL].to_numpy()

    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_group_sizes)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, group=valid_group_sizes, reference=lgb_train)

    params = {
        "objective": "lambdarank",
        "metric": ["ndcg"],
        "ndcg_eval_at": [10, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }

    print("Training LightGBM ranker (HIT-only)...")
    print("Train queries:", len(train_group_sizes), "| Valid queries:", len(valid_group_sizes))

    booster = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    booster.save_model(MODEL_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print("\nSaved HIT-only LightGBM ranker to:", MODEL_PATH)
    print("Saved features list to:", FEATURES_PATH)

if __name__ == "__main__":
    main()
