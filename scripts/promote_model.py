import argparse, os
from mlflow.tracking import MlflowClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--from-stage", required=True)
    ap.add_argument("--to-stage", required=True)
    args = ap.parse_args()

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    c = MlflowClient(tracking_uri=uri)

    versions = c.get_latest_versions(args.model, stages=[args.from_stage])
    if not versions:
        raise RuntimeError(f"No versions in stage {args.from_stage} for {args.model}")

    v = versions[0].version
    c.transition_model_version_stage(
        name=args.model,
        version=v,
        stage=args.to_stage,
        archive_existing_versions=True
    )
    print(f"Promoted {args.model} v{v}: {args.from_stage} -> {args.to_stage}")

if __name__ == "__main__":
    main()
