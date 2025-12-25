import os
import pandas as pd

IN_PATH = "../data/processed/ratings_clean.parquet"
OUT_DIR = "../data/processed"
OUT_PATH = os.path.join(OUT_DIR, "positive_interactions.parquet")


POSITIVE_RATING_THRESHOLD = 4.0

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # load cleaned ratings
    ratings = pd.read_parquet(IN_PATH)
    
    # Keep only positives 
    positives = ratings[ratings["rating"] >= POSITIVE_RATING_THRESHOLD].copy()
    
    # keep only what we need for retrival
    positives = positives[["userId", "movieId", "timestamp"]]
    
    # sort and remove duplicates
    positives = positives.sort_values(["userId","movieId","timestamp"])
    positives = positives.drop_duplicates(subset = ["userId","movieId"], keep = "last")
    
    positives.to_parquet(OUT_PATH, index=False)
    
    print("=== Step 2 done ===")
    print("Saved:", OUT_PATH)
    print("Positive interactions:", f"{len(positives):,}")
    print("Unique users:", f"{positives['userId'].nunique():,}")
    print("Unique movies:", f"{positives['movieId'].nunique():,}")
    print("Timestamp range:", positives["timestamp"].min(), "->", int(positives["timestamp"].max()))
    
    
if __name__ == "__main__":
    main()
    
    