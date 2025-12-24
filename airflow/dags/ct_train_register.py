from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

DEFAULT_ARGS = {"owner": "mlops", "retries": 0}

with DAG(
    dag_id="ct_train_register",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "ct"],
) as dag:

    download = BashOperator(
        task_id="download_kaggle",
        bash_command="python -m src.data.download_kaggle",
    )

    prep = BashOperator(
        task_id="prepare_data",
        bash_command="python -m src.data.prepare_sentiment140",
    )

    train = BashOperator(
        task_id="train_register",
        bash_command=(
            "python -m src.train.train_register_mlflow "
            "--branch '{{ dag_run.conf.get(\"branch\", \"dev\") }}' "
            "--git-sha '{{ dag_run.conf.get(\"git_sha\", \"local\") }}'"
        ),
    )

    download >> prep >> train
