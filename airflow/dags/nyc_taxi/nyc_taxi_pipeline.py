from airflow.sdk import Param, dag, task
from pendulum import datetime


@dag(
    dag_id="nyc_taxi_training",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops-zoomcamp", "homework"],
    params={
        "year": Param(2023, type="integer"),
        "month": Param(3, type="integer"),
    },
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
    def train_for_month(year: str, month: str) -> str:
        from nyc_taxi.nyc_taxi_training import run

        # Jinja-templated params arrive as strings here, so cast them
        return run(int(year), int(month))

    train_for_month(
        year="{{ params.year }}",
        month="{{ params.month }}",
    )


nyc_taxi_training()
