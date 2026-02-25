
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from prefect import task

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog()

mlflow.set_experiment("nyc-taxi-experiment")

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


@task(log_prints=True)
def train_model(df):
    print(f"Training model with {len(df)} rows.")
    mlflow.set_tag("model", "linear_regression")
    with mlflow.start_run(nested=True):
        features = ["PULocationID", "DOLocationID"]
        # Train data dictionaries - with ids as strings for one-hot encoding
        train_dicts = df[features].astype(str).to_dict(orient = 'records')
        dc = DictVectorizer()
        X_train = dc.fit_transform(train_dicts)
        y_train = df['duration'].values
        lr = LinearRegression()
        
        lr.fit(X_train, y_train)
        
        y_pred = lr.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        mlflow.sklearn.log_model(lr, artifact_path="models")
    return dc, lr


@task(log_prints=True)
def register_model(df):
   client = MlflowClient()
   experiment = client.get_experiment_by_name("nyc-taxi-experiment")
   run_id = client.search_runs(experiment.experiment_id,
                               run_view_type=ViewType.ACTIVE_ONLY,
                               max_results=1,
                               order_by=["metrics.rmse ASC"])[0].info.run_id
   mlflow.register_model(f"runs:/{run_id}/models", "nyc-taxi-model")
   

# use Perfect model to predict
# Read the data
dfs = pd.read_parquet("data/yellow_tripdata_2023-03.parquet")
# Number of observations
print(f"Number of rows: {len(dfs)}.")

df = read_dataframe("data/yellow_tripdata_2023-03.parquet")

print(f"Number of rows: {len(df)}.")
mlflow.end_run() 
_, model = train_model(df)
# Print the obtained run experiment id
print(f"Model Intercept: {round(model.intercept_, 2)}")
# register the model
register_model(df)



