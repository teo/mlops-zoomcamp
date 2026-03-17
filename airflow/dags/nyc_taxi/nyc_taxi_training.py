import pickle
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("https://mlflow.teom.org")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)


def read_dataframe(year: int, month: int) -> pd.DataFrame:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    print(f"Reading the data from {url}...")
    df = pd.read_parquet(url)

    print(len(df))

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df["duration"].apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


def create_X(df: pd.DataFrame, dv: DictVectorizer | None = None):
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv: DictVectorizer) -> str:
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        print(f"Intercept: {lr.intercept_}")

        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("model", "LinearRegression")

        preprocessor_path = models_folder / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")

        return run.info.run_id


def run(year: int, month: int) -> str:
    print(f"Running training for year={year}, month={month}")

    df_train = read_dataframe(year, month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(next_year, next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id
