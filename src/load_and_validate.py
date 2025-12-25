import os
import pandas as pd

RAW_DIR = "../data/raw/ml-20m"
OUT_DIR = "../data/processed"

os.makedirs(OUT_DIR, exist_ok=True)

ratings = pd.read_csv(f"{RAW_DIR}/rating.csv",
                      usecols = ["userId", "movieId", "rating", "timestamp"]
                      )

ratings = ratings.dropna()
ratings = ratings.drop_duplicates()

movies = pd.read_csv(f"{RAW_DIR}/movie.csv",
                     usecols=["movieId", "title", "genres"]
                     )
movies = movies.dropna()
movies = movies.drop_duplicates()

print("Ratings count:", len(ratings))
print("Unique users:", ratings["userId"].nunique())
print("unique movies:", ratings["movieId"].nunique())
print("Unique movies in movies file:", movies["movieId"].nunique())

print("Rating range:", ratings["rating"].min(), ratings["rating"].max())
print("Timestamp range:", ratings["timestamp"].min(), ratings["timestamp"].max())

ratings.to_parquet(f"{OUT_DIR}/ratings_clean.parquet", index=False)
movies.to_parquet(f"{OUT_DIR}/movies_clean.parquet", index=False)

