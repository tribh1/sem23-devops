import os, subprocess, pathlib

def ensure_kaggle_creds():
    # Kaggle CLI reads ~/.kaggle/kaggle.json (permission 600)
    kaggle_dir = pathlib.Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "kaggle.json"

    # Option A: provide as env var KAGGLE_JSON (full file content)
    kaggle_json = os.getenv("KAGGLE_JSON", "").strip()
    if kaggle_json and not token_path.exists():
        token_path.write_text(kaggle_json, encoding="utf-8")
        os.chmod(token_path, 0o600)

    if not token_path.exists():
        raise RuntimeError(
            "Missing Kaggle credentials. Provide ~/.kaggle/kaggle.json "
            "or env KAGGLE_JSON (content of kaggle.json)."
        )

def download_sentiment140(out_dir: str = "data/raw"):
    ensure_kaggle_creds()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Dataset: kazanova/sentiment140 :contentReference[oaicite:1]{index=1}
    cmd = ["kaggle", "datasets", "download", "-d", "kazanova/sentiment140", "-p", out_dir]
    subprocess.check_call(cmd)

    # unzip
    import zipfile
    zips = list(pathlib.Path(out_dir).glob("*.zip"))
    if not zips:
        raise RuntimeError("Download succeeded but no zip found in data/raw.")
    with zipfile.ZipFile(zips[0], "r") as zf:
        zf.extractall(out_dir)

if __name__ == "__main__":
    download_sentiment140()
    print("Downloaded & extracted Sentiment140 into data/raw")
