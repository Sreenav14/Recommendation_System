import os
import pandas as pd

from make_positives import OUT_DIR

IN_PATH = "../data/processed/positive_interactions.parquet"
OUT_PATH = "../data/processed"

# v1 : last 1 interaction test, second last interaction validation, rest training

TEST_ITEMS_PER_USER = 1
VAL_ITEMS_PER_USER = 1

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    df = pd.read_parquet(IN_PATH)
    
    # sort the items latest -> newest
    df = df.sort_values(["userId","timestamp"])
    
    train_parts = []
    val_parts = []
    test_parts = []
    
    # split per user
    for user_id, g in df.groupby("userId", sort = False):
        n = len(g)
        
        # if user has less than 2 interactions
        if n <= (TEST_ITEMS_PER_USER + VAL_ITEMS_PER_USER):
            train_parts.append(g)
            continue
        
        test = g.tail(TEST_ITEMS_PER_USER)
        val = g.iloc[-(TEST_ITEMS_PER_USER + VAL_ITEMS_PER_USER):-TEST_ITEMS_PER_USER]
        train = g.iloc[:-(TEST_ITEMS_PER_USER + VAL_ITEMS_PER_USER)]
        
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
        
    train_df = pd.concat(train_parts, ignore_index = True)
    val_df = pd.concat(val_parts, ignore_index = True) if val_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index = True) if test_parts else pd.DataFrame(columns=df.columns)
    
    # save
    train_path = os.path.join(OUT_DIR, "train.parquet")
    val_path = os.path.join(OUT_DIR, "val.parquet")
    test_path = os.path.join(OUT_DIR, "test.parquet")
    
    
    train_df.to_parquet(train_path, index = False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    
    # print stats
    print("=== Step 3 done: time-based split per user ===")
    print("Train rows:", f"{len(train_df):,}")
    print("Val rows:", f"{len(val_df):,}")
    print("Test rows:", f"{len(test_df):,}")
    
    print("Users in train:", f"{train_df['userId'].nunique():,}")
    print("Users in val:", f"{val_df['userId'].nunique():,}")
    print("Users in test:", f"{test_df['userId'].nunique():,}")

    # Quick sanity: latest timestamp should be mostly in test
    print("Train timestamp max:", int(train_df["timestamp"].max()))
    if len(val_df) > 0:
        print("Val timestamp max:", int(val_df["timestamp"].max()))
    if len(test_df) > 0:
        print("Test timestamp max:", int(test_df["timestamp"].max()))

if __name__ == "__main__":
    main()
    
    
        
             
    
    