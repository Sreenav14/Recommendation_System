from typing import Any


import os
import json
import pandas as pd

MOVIES_PATH = "../data/processed/movies_clean.parquet"
OUT_PATH = "../data/processed/movie_genre_features.json"

def main():
    movies = pd.read_parquet(MOVIES_PATH, columns=["movieId","genres"])
    movies["movieId"] = movies["movieId"].astype(str)
    movies["genres"] = movies["genres"].fillna("")
    
    # collect unique genres
    all_genres = set()
    for g in movies["genres"].to_list():
        for genre in g.split("|"):
            if genre and genre != "(no genre listed)":
                all_genres.add(genre)
                
    genre_list = sorted(all_genres)
    genre_to_idx = {g : i for i, g in enumerate(genre_list)}
    
    # build multi-hot vectors
    movie_to_vec = {}
    for mid, g in zip(movies["movieId"], movies["genres"]):
            vec = [0] * len(genre_list)
            for genre in g.split("|"):
                if genre and genre_to_idx :
                    vec[genre_to_idx[genre]] = 1
            movie_to_vec[mid] = vec
                
    
    out = {
        "genre_list" : genre_list,
        "movie_to_vec" : movie_to_vec
    }
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok = True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
        
        
    print("Done")
    print("saved", OUT_PATH)
    print("genres", len(genre_list))
    print("movies", len(movie_to_vec))
    

if __name__ == "__main__":
    main()
    
                     
            