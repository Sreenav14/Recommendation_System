import os
import json
from re import T
import pandas as pd
import numpy as np
import lightgbm as lgb

DATA_PATH = "../data/processed/ranker_dataset.parquet"
OUT_DIR = "../artifact/ranker_lgbm"
os.makedirs(OUT_DIR, exist_ok = True)

SEED = 42

def main():
    print("Loading ranker dataset...")
    df = pd.read_parquet(DATA_PATH)
    
    # sort by group so lightgbm understands
    df = df.sort_values("group").reset_index(drop = True)
    
    # features we have
    feature_cols = ["retrieval_score", "pop_rank"]
    
    X = df[feature_cols].values
    y = df["label"].values
    
    # group_size : each group has 200 rows
    group_sizes = df.groupby("group").size().values
    
    print("ROWS:", len(df))
    print("GROUPS:", len(group_sizes))
    print("Avg group size:", np.mean(group_sizes))
    print("Positives:", int(df["label"].sum()))
    
    #  Train/ Valid split by groups 
    n_groups = len(group_sizes)
    n_train_groups = int(n_groups * 0.8)
    
    train_mask = df["group"] < n_train_groups
    valid_mask = ~train_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    
    train_group = df.loc[train_mask].groupby("group").size().values
    valid_group = df.loc[valid_mask].groupby("group").size().values
    
    train_data = lgb.Dataset(X_train, label = y_train, group = train_group)
    valid_data = lgb.Dataset(X_valid, label = y_valid, group = valid_group)
    
    params = {
        "objective" : "lambdarank",
        "metric" : "ndcg",
        "ndcg_eval_at" : [10, 50],
        "learning_rate" : 0.05,
        "num_leaves" : 63,
        "min_data_in_leaf" : 100,
        "feature_fraction" : 0.9,
        "bagging_fraction" : 0.9,
        "bagging_freq" : 1,
        "verbosity" : -1,
        "seed" : SEED,
    }
    
    print("Training LightGBM ranker...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round = 100,
        valid_sets = [train_data, valid_data],
        valid_names = ["train", "valid"],
        callbacks=[lgb.early_stopping(50)]
    )
    
    model_path = os.path.join(OUT_DIR, "lgbm_ranker.txt")
    model.save_model(model_path)
    
    with open(os.path.join(OUT_DIR, "features.json"), "w") as f:
        json.dump("feature_cols", f)
        
    print("\n Saved LightGBM ranker to: ", model_path)
    print("Features:", feature_cols)
    print("saved features list to:", os.path.join(OUT_DIR, "features.json"))
    
    
if __name__ == "__main__":
    main()