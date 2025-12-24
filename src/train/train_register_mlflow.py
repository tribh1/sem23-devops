import argparse, os, json, time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/sentiment140.parquet")
    ap.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    ap.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "sentiment140"))
    ap.add_argument("--registered-model", default=os.getenv("REGISTERED_MODEL_NAME", "sentiment140-logreg"))
    ap.add_argument("--branch", default=os.getenv("GIT_BRANCH", "dev"))
    ap.add_argument("--git-sha", default=os.getenv("GIT_SHA", "local"))
    ap.add_argument("--max-rows", type=int, default=200000)  # speed knob
    ap.add_argument("--min-acc", type=float, default=0.80)
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    df = pd.read_parquet(args.data)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=200000)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=1)),
    ])

    t0 = time.time()
    with mlflow.start_run(run_name=f"{args.branch}-{args.git_sha[:7]}"):
        mlflow.log_params({
            "model_type": "logreg_tfidf",
            "branch": args.branch,
            "git_sha": args.git_sha,
            "max_rows": args.max_rows,
            "ngram_range": "(1,2)",
            "max_features": 200000,
        })

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        m = metrics(y_val, preds)
        for k, v in m.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("train_seconds", time.time() - t0)

        # register model
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=args.registered_model,
        )

        # gate
        if m["accuracy"] < args.min_acc:
            raise RuntimeError(f"Model gate failed: accuracy {m['accuracy']:.4f} < {args.min_acc}")

        print(json.dumps({"event":"train_ok","metrics":m,"registered_model":args.registered_model}))

if __name__ == "__main__":
    main()
