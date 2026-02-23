from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
import mlflow
import pandas as pd
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from feature_store import SimpleFeatureStore

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'ml_pipeline_with_feature_store',
    default_args=default_args,
    description='ML pipeline with feature store and MLflow tracking',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

def extract_and_load_data(**context):
    """Extract data and load into feature store"""
    print("Starting data extraction and feature store loading...")
    
    # Initialize feature store
    feature_store = SimpleFeatureStore()
    
    # Load iris dataset
    X, y = datasets.load_iris(return_X_y=True)
    iris_features_names = datasets.load_iris().feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Store training features
    train_entity_ids = [f"sample_{i}" for i in range(len(X_train))]
    train_feature_data = {
        "sepal_length": X_train[:, 0].tolist(),
        "sepal_width": X_train[:, 1].tolist(), 
        "petal_length": X_train[:, 2].tolist(),
        "petal_width": X_train[:, 3].tolist(),
        "target": y_train.tolist()
    }
    
    feature_descriptions = {
        "sepal_length": "Length of sepal in cm",
        "sepal_width": "Width of sepal in cm", 
        "petal_length": "Length of petal in cm",
        "petal_width": "Width of petal in cm",
        "target": "Iris species class"
    }
    
    feature_store.store_features(train_entity_ids, train_feature_data, feature_descriptions)
    
    # Store test features (without target)
    test_entity_ids = [f"test_sample_{i}" for i in range(len(X_test))]
    test_feature_data = {
        "sepal_length": X_test[:, 0].tolist(),
        "sepal_width": X_test[:, 1].tolist(), 
        "petal_length": X_test[:, 2].tolist(),
        "petal_width": X_test[:, 3].tolist(),
    }
    
    feature_store.store_features(test_entity_ids, test_feature_data)
    
    # Push data info to XCom for next tasks
    context['ti'].xcom_push(key='train_entity_ids', value=train_entity_ids)
    context['ti'].xcom_push(key='test_entity_ids', value=test_entity_ids)
    context['ti'].xcom_push(key='y_test', value=y_test.tolist())
    
    print(f"Stored {len(train_entity_ids)} training and {len(test_entity_ids)} test samples")
    return len(train_entity_ids), len(test_entity_ids)

def train_model(**context):
    """Train ML model using features from feature store"""
    print("Starting model training...")
    
    # Initialize MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.sklearn.autolog()
    mlflow.set_experiment("iris_logistic_regression_airflow")
    
    # Pull data from XCom
    ti = context['ti']
    train_entity_ids = ti.xcom_pull(task_ids='extract_and_load_data', key='train_entity_ids')
    test_entity_ids = ti.xcom_pull(task_ids='extract_and_load_data', key='test_entity_ids')
    
    # Initialize feature store
    feature_store = SimpleFeatureStore()
    
    # Retrieve training features
    train_features = feature_store.get_features(
        train_entity_ids, 
        ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    )
    
    # Prepare training data
    X_train = train_features[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y_train = train_features["target"].values
    
    # Train model
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 42
    }
    
    with mlflow.start_run() as run:
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            registered_model_name="iris_with_feature_store_airflow"
        )
        
        # Push run ID for next tasks
        context['ti'].xcom_push(key='mlflow_run_id', value=run.info.run_id)
    
    print("Model training completed")
    return run.info.run_id

def evaluate_model(**context):
    """Evaluate model performance"""
    print("Starting model evaluation...")
    
    # Pull data from XCom
    ti = context['ti']
    test_entity_ids = ti.xcom_pull(task_ids='extract_and_load_data', key='test_entity_ids')
    y_test = ti.xcom_pull(task_ids='extract_and_load_data', key='y_test')
    mlflow_run_id = ti.xcom_pull(task_ids='train_model', key='mlflow_run_id')
    
    # Initialize feature store and MLflow
    feature_store = SimpleFeatureStore()
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Retrieve test features
    test_features = feature_store.get_features(
        test_entity_ids, 
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    
    X_test = test_features[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    
    # Load model from MLflow
    model_uri = f"runs:/{mlflow_run_id}/iris_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Make predictions
    y_pred = loaded_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Log metrics to MLflow
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_r2": r2,
            "test_mse": mse
        })
    
    # Store predictions in feature store
    prediction_data = {
        "predicted_target": y_pred.tolist()
    }
    feature_store.store_features(test_entity_ids, prediction_data)
    
    print(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy, "f1": f1}

def validate_pipeline(**context):
    """Validate pipeline results and send notifications"""
    print("Starting pipeline validation...")
    
    # Pull metrics from previous task
    ti = context['ti']
    metrics = ti.xcom_pull(task_ids='evaluate_model')
    
    accuracy = metrics['accuracy']
    f1 = metrics['f1']
    
    # Validation criteria
    if accuracy >= 0.9 and f1 >= 0.9:
        status = "SUCCESS"
        message = f"✅ Pipeline validation passed! Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
    else:
        status = "WARNING"
        message = f"⚠️ Pipeline validation warning! Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
    
    print(message)
    
    # You could add Slack notification here
    # SlackWebhookOperator(
    #     task_id='slack_notification',
    #     webhook_token='your-slack-webhook-token',
    #     message=message,
    #     dag=dag
    # ).execute(context=context)
    
    return {"status": status, "message": message}

# Define tasks
extract_task = PythonOperator(
    task_id='extract_and_load_data',
    python_callable=extract_and_load_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_pipeline',
    python_callable=validate_pipeline,
    dag=dag,
)

# Set task dependencies
extract_task >> train_task >> evaluate_task >> validate_task