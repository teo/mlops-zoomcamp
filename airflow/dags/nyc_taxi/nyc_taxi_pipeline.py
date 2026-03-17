from airflow.sdk import dag, task
from pendulum import datetime


@dag(
    dag_id="nyc_taxi_training",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops-zoomcamp", "homework"],
)
def nyc_taxi_training():
    @task.virtualenv(
        task_id="train_for_month",
        requirements=[
            "mlflow-skinny",
            "pandas",
            "pyarrow",
            "scikit-learn",
        ],
        system_site_packages=False,
    )
    def train_for_month(year: int, month: int) -> str:
        from nyc_taxi.nyc_taxi_training import run

        return run(year, month)

    train_for_month(year=2023, month=3)


nyc_taxi_training()
