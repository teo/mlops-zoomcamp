from airflow.sdk import dag, task
from pendulum import datetime

from nyc_taxi.nyc_taxi_training import run


@task
def train_for_month(year: int, month: int) -> str:
    return run(year, month)


@dag(
    dag_id="nyc_taxi_training",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops-zoomcamp", "homework"],
)
def nyc_taxi_training():
    train_for_month(year=2023, month=3)


nyc_taxi_training()
