import os
import pandas as pd

TRAIN_PATH = "../data/processed/train.parquet"
VAL_PATH   = "../data/processed/val.parquet"

OUT_TRAIN = "../data/processed/train_hist.parquet"
OUT_VAL   = "../data/processed/val_hist.parquet"

HIST_N = 20

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)
    if "timestamp" in df.columns:
        df = df.sort_values(["userId", "timestamp"])
    else:
        df = df.sort_values(["userId"])
    return df

def build_train_hist(train_df: pd.DataFrame, hist_n: int) -> pd.DataFrame:
    train_df = _ensure_cols(train_df)

    users, targets, histories = [], [], []
    for user, g in train_df.groupby("userId", sort=False):
        movies = g["movieId"].tolist()
        for i in range(len(movies)):
            hist = movies[max(0, i - hist_n): i]
            if not hist:
                continue
            users.append(user)
            targets.append(movies[i])
            histories.append(hist)

    return pd.DataFrame({"userId": users, "movieId": targets, "hist_movieIds": histories})

def build_val_hist(train_df: pd.DataFrame, val_df: pd.DataFrame, hist_n: int) -> pd.DataFrame:
    """
    For each user:
      history comes from the end of train interactions
      target is the user's val interaction(s)
    """
    train_df = _ensure_cols(train_df)
    val_df = _ensure_cols(val_df)

    train_by_user = train_df.groupby("userId")["movieId"].apply(list).to_dict()

    users, targets, histories = [], [], []
    for user, g in val_df.groupby("userId", sort=False):
        val_movies = g["movieId"].tolist()
        past = train_by_user.get(user, [])

        # as we go through val targets, we extend the history with previous val targets too
        hist = past[-hist_n:]  # last N from train
        for m in val_movies:
            if len(hist) == 0:
                # if user has zero past history at all, skip
                # (rare if your split was correct)
                hist = [m]
                continue
            users.append(user)
            targets.append(m)
            histories.append(hist[-hist_n:])

            # update history by appending this val movie (so next val movie has more history)
            hist = hist + [m]

    return pd.DataFrame({"userId": users, "movieId": targets, "hist_movieIds": histories})

def main():
    print("Loading train/val...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)

    print("Building train_hist...")
    train_hist = build_train_hist(train_df, HIST_N)
    print("Train_hist shape:", train_hist.shape)

    print("Building val_hist (using train history)...")
    val_hist = build_val_hist(train_df, val_df, HIST_N)
    print("Val_hist shape:", val_hist.shape)

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    train_hist.to_parquet(OUT_TRAIN, index=False)
    val_hist.to_parquet(OUT_VAL, index=False)

    print("\nSaved:")
    print(" ", OUT_TRAIN)
    print(" ", OUT_VAL)

    print("\nSample val rows:")
    print(val_hist.head(3))

if __name__ == "__main__":
    main()
