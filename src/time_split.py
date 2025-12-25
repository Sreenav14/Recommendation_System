import os
import pandas as pd

from make_positives import OUT_DIR

IN_PATH = "../data/processed/positive_interactions.parquet"
OUT_PATH = "../data/processed"

# v1 : last 1 interaction test, second last interaction validation, rest training

TEST_ITEMS_PER_USER = 1
VAL_ITEMS_PER_USER = 1

def main():
    os.makedir(OUT_DIR, exist_ok=True)
    
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
            
             
    
    