import pytest
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

class TestMLPipeline:
    """Test suite for ML pipeline components"""
    
    @pytest.fixture
    def sample_data(self):
        """Load sample iris dataset for testing"""
        X, y = datasets.load_iris(return_X_y=True)
        return X, y
    
    @pytest.fixture
    def sample_dataframe(self, sample_data):
        """Create sample DataFrame for testing"""
        X, y = sample_data
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        return df
    
    def test_data_loading(self, sample_data):
        """Test data loading functionality"""
        X, y = sample_data
        assert X is not None
        assert y is not None
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 4  # 4 features for iris
        
    def test_data_validation(self, sample_dataframe):
        """Test data validation functionality"""
        from main import DataQualityValidator
        
        validator = DataQualityValidator()
        results = validator.validate_dataset(sample_dataframe, "test_dataset")
        
        assert results is not None
        assert "row_count" in results
        assert "column_count" in results
        assert results["row_count"] > 0
        assert results["column_count"] > 0
        
    def test_feature_store_operations(self, sample_dataframe):
        """Test feature store operations"""
        from main import SimpleFeatureStore
        
        feature_store = SimpleFeatureStore(":memory:")  # Use in-memory DB for testing
        
        # Test data preparation
        entity_ids = ["test_entity_1", "test_entity_2"]
        feature_data = {
            "sepal_length": [5.1, 6.2],
            "sepal_width": [3.5, 2.9],
            "petal_length": [1.4, 4.3],
            "petal_width": [0.2, 1.3]
        }
        
        # Test storing features
        feature_store.store_features(entity_ids, feature_data)
        
        # Test retrieving features
        retrieved_features = feature_store.get_features(
            entity_ids, 
            ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        )
        
        assert len(retrieved_features) == 2
        assert "sepal_length" in retrieved_features.columns
        
    def test_model_training(self, sample_data):
        """Test model training with MLflow"""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        with mlflow.start_run() as run:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # Log to MLflow
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.sklearn.log_model(model, "test_model")
        
        # Assertions
        assert accuracy > 0.7  # Basic accuracy threshold
        assert run.info.run_id is not None
        assert run.info.status == "FINISHED"
        
    def test_model_loading(self, sample_data):
        """Test model loading from MLflow"""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train and log model
        with mlflow.start_run() as run:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "test_model")
        
        # Load model
        model_uri = f"runs:/{run.info.run_id}/test_model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Test predictions
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1, 2] for pred in predictions)  # Iris classes
        
    def test_network_analysis(self, sample_dataframe):
        """Test network analysis functionality"""
        from main import NetworkAnalyzer
        
        analyzer = NetworkAnalyzer()
        
        # Build correlation network
        network = analyzer.build_feature_correlation_network(sample_dataframe, threshold=0.3)
        
        # Get network metrics
        metrics = analyzer.get_network_metrics()
        
        assert network is not None
        assert "node_count" in metrics
        assert "edge_count" in metrics
        assert metrics["node_count"] > 0
        
    def test_infrastructure_health_check(self):
        """Test infrastructure health checking"""
        from main import InfrastructureHealthChecker, DeploymentConfig, Environment
        
        config = DeploymentConfig(
            environment=Environment.DEVELOPMENT,
            mlflow_tracking_uri="http://localhost:5000",
            spark_master="local[*]",
            kafka_bootstrap_servers="localhost:9092",
            redis_host="localhost",
            redis_port=6379,
            feature_store_path="test.db",
            model_registry_uri="sqlite:///test.db",
            monitoring_enabled=True,
            log_level="INFO"
        )
        
        health_checker = InfrastructureHealthChecker(config)
        
        # Test individual health checks
        mlflow_status = health_checker.check_mlflow_health()
        redis_status = health_checker.check_redis_health()
        disk_status = health_checker.check_disk_space()
        
        assert "service" in mlflow_status
        assert "status" in mlflow_status
        assert "timestamp" in mlflow_status
        
        assert "service" in redis_status
        assert "status" in redis_status
        
        assert "service" in disk_status
        assert "free_percent" in disk_status
        
    def test_environment_configuration(self):
        """Test environment configuration management"""
        from main import EnvironmentManager, Environment
        
        # Test with default config (should not fail)
        env_manager = EnvironmentManager("config/deployment.yaml")
        config = env_manager._get_default_config(Environment.DEVELOPMENT)
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.mlflow_tracking_uri is not None
        assert config.spark_master is not None
        assert config.kafka_bootstrap_servers is not None
        
    def test_automated_testing_pipeline(self, sample_dataframe):
        """Test automated testing pipeline"""
        from main import AutomatedTestingPipeline, DeploymentConfig, Environment
        
        config = DeploymentConfig(
            environment=Environment.DEVELOPMENT,
            mlflow_tracking_uri="http://localhost:5000",
            spark_master="local[*]",
            kafka_bootstrap_servers="localhost:9092",
            redis_host="localhost",
            redis_port=6379,
            feature_store_path="test.db",
            model_registry_uri="sqlite:///test.db",
            monitoring_enabled=True,
            log_level="INFO"
        )
        
        testing_pipeline = AutomatedTestingPipeline(config)
        
        # Test individual test methods
        data_test = testing_pipeline._test_data_pipeline()
        feature_store_test = testing_pipeline._test_feature_store()
        mlflow_test = testing_pipeline._test_mlflow_integration()
        
        assert "success" in data_test
        assert "success" in feature_store_test
        assert "success" in mlflow_test

if __name__ == "__main__":
    # Parse command line arguments for standalone testing
    parser = argparse.ArgumentParser(description="Run ML pipeline tests")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--test-data", help="Path to test data file")
    
    args = parser.parse_args()
    
    # Run tests
    pytest.main([__file__, "-v"])
