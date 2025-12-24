import os, pathlib
import pandas as pd

COLS = ["target", "ids", "date", "flag", "user", "text"]

def load_raw_csv(raw_dir="data/raw") -> pd.DataFrame:
    raw_dir = pathlib.Path(raw_dir)
    # Kaggle zip thường chứa file training.1600000.processed.noemoticon.csv (227MB)
    # Không commit file này vào Git.
    candidates = list(raw_dir.glob("training.*.csv"))
    if not candidates:
        # fallback: any csv
        candidates = list(raw_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {raw_dir}. Did you download dataset?")

    path = candidates[0]
    df = pd.read_csv(
        path,
        encoding="latin-1",
        header=None,
        names=COLS,
    )
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    # keep only negative(0) and positive(4); map to 0/1
    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = (df["target"] == 4).astype(int)
    df["text"] = df["text"].astype(str)
    return df[["text", "label"]]

def save(df: pd.DataFrame, out_path="data/processed/sentiment140.parquet"):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    df = load_raw_csv()
    df2 = transform(df)
    save(df2)
    print("Saved processed dataset to data/processed/sentiment140.parquet")
