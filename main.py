import os
# Set Python environment variables for Spark
os.environ['PYSPARK_PYTHON'] = 'python3.10'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3.10'

import mlflow
from mlflow import models
import pandas as pd
import json
import threading
import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler,StandardScaler, PCA
from pyspark.ml import Pipeline as SparkPipeline
    
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from feature_store import SimpleFeatureStore

# Stanford-inspired tools
import networkx as nx
import dask.dataframe as dd
import dask.distributed as distributed
from great_expectations.dataset import PandasDataset
import redis
import psutil
import time
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import warnings

# MLOps and Infrastructure tools
import yaml
import os
from dotenv import load_dotenv
import git
import docker
import kubernetes
import requests
import structlog
from pathlib import Path
import subprocess
import socket
import urllib.parse
from dataclasses import dataclass
from enum import Enum
import tempfile
import shutil

# Testing and CI/CD
import pytest
from unittest.mock import Mock, patch

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Kafka imports
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    print("Warning: kafka-python not installed. Install with: pip install kafka-python")
    KAFKA_AVAILABLE = False

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.sklearn.autolog()

spark = SparkSession.builder \
    .appName("ML_Feature_Engineering") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Initialize feature store
feature_store = SimpleFeatureStore()

# Initialize Redis for caching
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis connected successfully")
except Exception as e:
    print(f"Redis not available: {e}")
    REDIS_AVAILABLE = False

# Initialize Dask client for distributed computing
try:
    dask_client = distributed.Client()
    DASK_AVAILABLE = True
    print(f"Dask client started: {dask_client.dashboard_link}")
except Exception as e:
    print(f"Dask not available: {e}")
    DASK_AVAILABLE = False

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
FEATURE_UPDATES_TOPIC = 'feature-updates'
MODEL_PREDICTIONS_TOPIC = 'model-predictions'

class DataQualityValidator:
    """Stanford-inspired data quality validation using Great Expectations"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """Validate dataset with comprehensive quality checks"""
        ge_df = PandasDataset(df)
        
        validation_results = {
            "dataset_name": dataset_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # Basic statistical validation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            validation_results["statistical_summary"] = df[numeric_columns].describe().to_dict()
        
        # Cache results in Redis if available
        if REDIS_AVAILABLE:
            cache_key = f"data_quality:{dataset_name}:{int(time.time())}"
            redis_client.setex(cache_key, 3600, json.dumps(validation_results, default=str))
        
        self.validation_results[dataset_name] = validation_results
        return validation_results
    
    def check_data_drift(self, current_df: pd.DataFrame, reference_df: pd.DataFrame, 
                        columns: List[str] = None) -> Dict[str, Any]:
        """Check for data drift between current and reference datasets"""
        if columns is None:
            columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
        
        drift_results = {}
        for col in columns:
            if col in current_df.columns and col in reference_df.columns:
                current_mean = current_df[col].mean()
                ref_mean = reference_df[col].mean()
                drift_pct = abs((current_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0
                
                drift_results[col] = {
                    "reference_mean": ref_mean,
                    "current_mean": current_mean,
                    "drift_percentage": drift_pct,
                    "drift_detected": drift_pct > 10.0  # 10% threshold
                }
        
        return drift_results

class NetworkAnalyzer:
    """Stanford Network Analysis Platform (SNAP) inspired network analysis"""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def build_feature_correlation_network(self, df: pd.DataFrame, threshold: float = 0.5) -> nx.Graph:
        """Build network of correlated features"""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        # Create network from correlations
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > threshold:
                        self.graph.add_edge(col1, col2, weight=abs(corr_value), correlation=corr_value)
        
        return self.graph
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network analysis metrics"""
        if len(self.graph.nodes()) == 0:
            return {"error": "No network built yet"}
        
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "clustering_coefficient": nx.average_clustering(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
            "centrality": dict(nx.degree_centrality(self.graph))
        }

class DistributedProcessor:
    """Dask-based distributed processing for scalable data operations"""
    
    def __init__(self):
        self.client = dask_client if DASK_AVAILABLE else None
    
    def process_large_dataset(self, df: pd.DataFrame, operation: str = "feature_engineering") -> pd.DataFrame:
        """Process large datasets using Dask for scalability"""
        if not DASK_AVAILABLE:
            print("Dask not available, using pandas")
            return df
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=4)
        
        if operation == "feature_engineering":
            # Example distributed feature engineering
            ddf['feature_sum'] = ddf.select_dtypes(include=[np.number]).sum(axis=1)
            ddf['feature_mean'] = ddf.select_dtypes(include=[np.number]).mean(axis=1)
        
        # Compute and return to pandas
        return ddf.compute()
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get Dask cluster information"""
        if not DASK_AVAILABLE:
            return {"error": "Dask not available"}
        
        return {
            "dashboard_link": self.client.dashboard_link,
            "workers": len(self.client.scheduler_info()['workers']),
            "memory_available": sum(worker['memory_limit'] for worker in self.client.scheduler_info()['workers'].values()),
            "cores_available": sum(worker['nthreads'] for worker in self.client.scheduler_info()['workers'].values())
        }

class PerformanceMonitor:
    """System performance monitoring inspired by Stanford's infrastructure monitoring"""
    
    def __init__(self):
        self.metrics_history = []
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system performance metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            "process_count": len(psutil.pids())
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def log_metrics_to_mlflow(self, metrics: Dict[str, Any]):
        """Log performance metrics to MLflow"""
        mlflow.log_metrics({
            "system_cpu_percent": metrics["cpu_percent"],
            "system_memory_percent": metrics["memory_percent"],
            "system_disk_usage": metrics["disk_usage"]
        })

# Initialize Stanford-inspired components
data_validator = DataQualityValidator()
network_analyzer = NetworkAnalyzer()
distributed_processor = DistributedProcessor()
performance_monitor = PerformanceMonitor()

# MLOps and Infrastructure Components
class Environment(Enum):
    """Environment types for deployment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DeploymentConfig:
    """Deployment configuration data class"""
    environment: Environment
    mlflow_tracking_uri: str
    spark_master: str
    kafka_bootstrap_servers: str
    redis_host: str
    redis_port: int
    feature_store_path: str
    model_registry_uri: str
    monitoring_enabled: bool
    log_level: str

class InfrastructureHealthChecker:
    """Infrastructure health monitoring and troubleshooting"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.health_status = {}
        self.logger = logger.bind(component="InfrastructureHealthChecker")
    
    def check_mlflow_health(self) -> Dict[str, Any]:
        """Check MLflow tracking server health"""
        try:
            response = requests.get(f"{self.config.mlflow_tracking_uri}/health", timeout=10)
            status = {
                "service": "mlflow",
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            status = {
                "service": "mlflow",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.error("MLflow health check failed", error=str(e))
        
        self.health_status["mlflow"] = status
        return status
    
    def check_spark_health(self) -> Dict[str, Any]:
        """Check Spark cluster health"""
        try:
            # Check Spark master UI
            response = requests.get(f"{self.config.spark_master}/", timeout=10)
            status = {
                "service": "spark",
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            status = {
                "service": "spark",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.error("Spark health check failed", error=str(e))
        
        self.health_status["spark"] = status
        return status
    
    def check_kafka_health(self) -> Dict[str, Any]:
        """Check Kafka cluster health"""
        try:
            # Try to connect to Kafka bootstrap servers
            import kafka
            from kafka.admin import KafkaAdminClient
            
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                client_id='health_check'
            )
            
            # Get cluster metadata
            metadata = admin_client._client.cluster
            status = {
                "service": "kafka",
                "status": "healthy",
                "brokers": len(metadata.brokers()),
                "topics": len(metadata.topics()),
                "timestamp": datetime.now().isoformat()
            }
            admin_client.close()
        except Exception as e:
            status = {
                "service": "kafka",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.error("Kafka health check failed", error=str(e))
        
        self.health_status["kafka"] = status
        return status
    
    def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            import redis
            client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            
            # Ping Redis
            pong = client.ping()
            info = client.info()
            
            status = {
                "service": "redis",
                "status": "healthy" if pong else "unhealthy",
                "used_memory": info.get('used_memory_human', 'unknown'),
                "connected_clients": info.get('connected_clients', 0),
                "uptime_seconds": info.get('uptime_in_seconds', 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            status = {
                "service": "redis",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.error("Redis health check failed", error=str(e))
        
        self.health_status["redis"] = status
        return status
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        disk_usage = psutil.disk_usage('/')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        
        status = {
            "service": "disk_space",
            "status": "warning" if free_percent < 20 else "healthy",
            "total_gb": round(disk_usage.total / (1024**3), 2),
            "used_gb": round(disk_usage.used / (1024**3), 2),
            "free_gb": round(disk_usage.free / (1024**3), 2),
            "free_percent": round(free_percent, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        if free_percent < 10:
            status["status"] = "critical"
            self.logger.warning("Critical disk space", free_percent=free_percent)
        
        self.health_status["disk_space"] = status
        return status
    
    def run_full_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        self.logger.info("Starting full infrastructure health check")
        
        checks = [
            self.check_mlflow_health,
            self.check_spark_health,
            self.check_kafka_health,
            self.check_redis_health,
            self.check_disk_space
        ]
        
        results = {}
        for check in checks:
            try:
                result = check()
                results[result["service"]] = result
            except Exception as e:
                self.logger.error("Health check failed", check=check.__name__, error=str(e))
        
        # Overall status
        unhealthy_services = [s for s in results.values() if s.get("status") in ["error", "critical"]]
        overall_status = "unhealthy" if unhealthy_services else "healthy"
        
        summary = {
            "overall_status": overall_status,
            "unhealthy_services": [s["service"] for s in results.values() if s.get("status") in ["error", "critical"]],
            "total_checks": len(results),
            "timestamp": datetime.now().isoformat(),
            "detailed_results": results
        }
        
        self.logger.info("Health check completed", 
                        overall_status=overall_status,
                        unhealthy_count=len(unhealthy_services))
        
        return summary

class EnvironmentManager:
    """Environment configuration management"""
    
    def __init__(self, config_path: str = "config/deployment.yaml"):
        self.config_path = config_path
        self.logger = logger.bind(component="EnvironmentManager")
    
    def load_config(self, environment: Environment) -> DeploymentConfig:
        """Load configuration for specific environment"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            env_config = config_data.get(environment.value, {})
            
            config = DeploymentConfig(
                environment=environment,
                mlflow_tracking_uri=env_config.get('mlflow_tracking_uri', 'http://localhost:5000'),
                spark_master=env_config.get('spark_master', 'local[*]'),
                kafka_bootstrap_servers=env_config.get('kafka_bootstrap_servers', 'localhost:9092'),
                redis_host=env_config.get('redis_host', 'localhost'),
                redis_port=env_config.get('redis_port', 6379),
                feature_store_path=env_config.get('feature_store_path', 'feature_store.db'),
                model_registry_uri=env_config.get('model_registry_uri', 'sqlite:///mlflow.db'),
                monitoring_enabled=env_config.get('monitoring_enabled', True),
                log_level=env_config.get('log_level', 'INFO')
            )
            
            self.logger.info("Configuration loaded", environment=environment.value)
            return config
            
        except Exception as e:
            self.logger.error("Failed to load configuration", 
                            environment=environment.value, 
                            error=str(e))
            # Fallback to development defaults
            return self._get_default_config(environment)
    
    def _get_default_config(self, environment: Environment) -> DeploymentConfig:
        """Get default configuration for environment"""
        return DeploymentConfig(
            environment=environment,
            mlflow_tracking_uri="http://localhost:5000",
            spark_master="local[*]",
            kafka_bootstrap_servers="localhost:9092",
            redis_host="localhost",
            redis_port=6379,
            feature_store_path="feature_store.db",
            model_registry_uri="sqlite:///mlflow.db",
            monitoring_enabled=True,
            log_level="INFO"
        )
    
    def validate_config(self, config: DeploymentConfig) -> bool:
        """Validate configuration"""
        try:
            # Validate MLflow URI
            parsed_uri = urllib.parse.urlparse(config.mlflow_tracking_uri)
            if not parsed_uri.scheme in ['http', 'https']:
                self.logger.error("Invalid MLflow URI", uri=config.mlflow_tracking_uri)
                return False
            
            # Validate Redis connection
            try:
                client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    decode_responses=True
                )
                client.ping()
                client.close()
            except Exception as e:
                self.logger.error("Redis connection failed", 
                                host=config.redis_host, 
                                port=config.redis_port, 
                                error=str(e))
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error("Configuration validation failed", error=str(e))
            return False

class MLOpsManager:
    """MLOps processes and CI/CD deployment strategies"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logger.bind(component="MLOpsManager")
        self.git_repo = None
        self._initialize_git()
    
    def _initialize_git(self):
        """Initialize Git repository"""
        try:
            self.git_repo = git.Repo(search_parent_directories=True)
            self.logger.info("Git repository initialized", 
                            branch=self.git_repo.active_branch.name)
        except Exception as e:
            self.logger.error("Failed to initialize Git repository", error=str(e))
    
    def create_model_version_tag(self, model_name: str, version: str, 
                               metrics: Dict[str, float]) -> str:
        """Create Git tag for model version"""
        try:
            tag_name = f"{model_name}/v{version}"
            
            # Create annotated tag with metrics
            message = f"Model {model_name} version {version}\n\nMetrics:\n"
            for metric, value in metrics.items():
                message += f"{metric}: {value}\n"
            
            self.git_repo.create_tag(tag_name, message=message)
            self.logger.info("Model version tag created", 
                            model=model_name, 
                            version=version, 
                            tag=tag_name)
            
            return tag_name
            
        except Exception as e:
            self.logger.error("Failed to create model version tag", 
                            model=model_name, 
                            version=version, 
                            error=str(e))
            return None
    
    def run_automated_tests(self, model_path: str, test_data_path: str) -> Dict[str, Any]:
        """Run automated tests for model"""
        try:
            # Run pytest programmatically
            test_result = subprocess.run([
                'python', '-m', 'pytest', 
                'tests/test_model.py',
                '--model-path', model_path,
                '--test-data', test_data_path,
                '--junitxml', 'test-results.xml',
                '--cov=ml_model',
                '--cov-report=xml'
            ], capture_output=True, text=True)
            
            results = {
                "exit_code": test_result.returncode,
                "stdout": test_result.stdout,
                "stderr": test_result.stderr,
                "passed": test_result.returncode == 0,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("Automated tests completed", 
                            passed=results["passed"],
                            exit_code=results["exit_code"])
            
            return results
            
        except Exception as e:
            self.logger.error("Failed to run automated tests", error=str(e))
            return {"error": str(e), "passed": False}
    
    def deploy_model_staging(self, model_uri: str, stage: str = "Staging") -> Dict[str, Any]:
        """Deploy model to staging environment"""
        try:
            # Load model from MLflow
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Validate model
            validation_results = self._validate_model(model)
            if not validation_results["valid"]:
                return {"error": "Model validation failed", "details": validation_results}
            
            # Register model in MLflow Model Registry
            model_details = mlflow.register_model(
                model_uri,
                "iris_classifier",
                await_registration_for=30
            )
            
            # Transition model to staging
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="iris_classifier",
                version=model_details.version,
                stage=stage
            )
            
            deployment_info = {
                "model_name": "iris_classifier",
                "version": model_details.version,
                "stage": stage,
                "model_uri": model_uri,
                "deployment_time": datetime.now().isoformat(),
                "validation_results": validation_results
            }
            
            self.logger.info("Model deployed to staging", 
                            model_name="iris_classifier",
                            version=model_details.version,
                            stage=stage)
            
            return deployment_info
            
        except Exception as e:
            self.logger.error("Failed to deploy model to staging", error=str(e))
            return {"error": str(e)}
    
    def _validate_model(self, model) -> Dict[str, Any]:
        """Validate model before deployment"""
        try:
            # Basic validation checks
            validation_results = {
                "valid": True,
                "checks": []
            }
            
            # Check if model is callable
            if not hasattr(model, 'predict'):
                validation_results["valid"] = False
                validation_results["checks"].append("Model missing predict method")
            
            # Check model metadata
            if hasattr(model, 'metadata'):
                validation_results["checks"].append("Model metadata present")
            else:
                validation_results["checks"].append("Warning: No model metadata found")
            
            return validation_results
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def rollback_model(self, model_name: str, target_version: int) -> Dict[str, Any]:
        """Rollback model to previous version"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get current production model
            production_models = client.search_model_versions(
                f"name='{model_name}' and stage='Production'"
            )
            
            if not production_models:
                return {"error": "No production model found"}
            
            current_version = production_models[0].version
            
            # Archive current production model
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Archived"
            )
            
            # Promote target version to production
            client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production"
            )
            
            rollback_info = {
                "model_name": model_name,
                "previous_version": current_version,
                "rollback_version": target_version,
                "rollback_time": datetime.now().isoformat(),
                "success": True
            }
            
            self.logger.info("Model rollback completed", 
                            model_name=model_name,
                            from_version=current_version,
                            to_version=target_version)
            
            return rollback_info
            
        except Exception as e:
            self.logger.error("Model rollback failed", 
                            model_name=model_name, 
                            target_version=target_version, 
                            error=str(e))
            return {"error": str(e), "success": False}

class AutomatedTestingPipeline:
    """Automated testing and validation pipeline"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logger.bind(component="AutomatedTestingPipeline")
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        test_results = {
            "data_pipeline_test": self._test_data_pipeline(),
            "feature_store_test": self._test_feature_store(),
            "model_training_test": self._test_model_training(),
            "kafka_integration_test": self._test_kafka_integration(),
            "mlflow_integration_test": self._test_mlflow_integration()
        }
        
        overall_success = all(result.get("success", False) for result in test_results.values())
        
        summary = {
            "overall_success": overall_success,
            "test_results": test_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info("Integration tests completed", 
                        overall_success=overall_success)
        
        return summary
    
    def _test_data_pipeline(self) -> Dict[str, Any]:
        """Test data pipeline"""
        try:
            # Test data loading and validation
            X, y = datasets.load_iris(return_X_y=True)
            df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            
            # Test data validation
            validation_results = data_validator.validate_dataset(df, "test_dataset")
            
            success = validation_results["row_count"] > 0 and validation_results["column_count"] > 0
            
            return {
                "success": success,
                "row_count": validation_results["row_count"],
                "column_count": validation_results["column_count"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_feature_store(self) -> Dict[str, Any]:
        """Test feature store"""
        try:
            # Test feature store operations
            test_entity_ids = ["test_entity_1", "test_entity_2"]
            test_feature_data = {
                "feature1": [1.0, 2.0],
                "feature2": [3.0, 4.0]
            }
            
            feature_store.store_features(test_entity_ids, test_feature_data)
            retrieved_features = feature_store.get_features(test_entity_ids, ["feature1", "feature2"])
            
            success = len(retrieved_features) == 2
            
            return {
                "success": success,
                "entities_stored": len(test_entity_ids),
                "entities_retrieved": len(retrieved_features)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_model_training(self) -> Dict[str, Any]:
        """Test model training"""
        try:
            # Test model training with MLflow
            X, y = datasets.load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            with mlflow.start_run() as run:
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.sklearn.log_model(model, "test_model")
            
            success = accuracy > 0.8  # Basic accuracy threshold
            
            return {
                "success": success,
                "accuracy": accuracy,
                "run_id": run.info.run_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_kafka_integration(self) -> Dict[str, Any]:
        """Test Kafka integration"""
        if not KAFKA_AVAILABLE:
            return {"success": True, "message": "Kafka not available, test skipped"}
        
        try:
            # Test Kafka producer
            producer = KafkaFeatureProducer()
            test_message = {"test": "message", "timestamp": datetime.now().isoformat()}
            
            success = producer.publish_feature_update("test_entity", test_message)
            producer.close()
            
            return {
                "success": success,
                "message_published": success
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_mlflow_integration(self) -> Dict[str, Any]:
        """Test MLflow integration"""
        try:
            # Test MLflow tracking
            with mlflow.start_run() as run:
                mlflow.log_param("test_param", "test_value")
                mlflow.log_metric("test_metric", 0.95)
                
                # Test model logging
                model = LogisticRegression(max_iter=1000, random_state=42)
                mlflow.sklearn.log_model(model, "test_model")
            
            # Test model loading
            loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/test_model")
            
            success = loaded_model is not None
            
            return {
                "success": success,
                "run_id": run.info.run_id,
                "model_loaded": success
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize MLOps components
environment_manager = EnvironmentManager()

# Load configuration based on environment
ENVIRONMENT = Environment(os.getenv('ENVIRONMENT', 'development'))
deployment_config = environment_manager.load_config(ENVIRONMENT)

# Validate configuration
if environment_manager.validate_config(deployment_config):
    logger.info("Configuration validated successfully", environment=ENVIRONMENT.value)
else:
    logger.warning("Configuration validation failed, using defaults")

# Initialize MLOps components
health_checker = InfrastructureHealthChecker(deployment_config)
mlops_manager = MLOpsManager(deployment_config)
testing_pipeline = AutomatedTestingPipeline(deployment_config)

class KafkaFeatureProducer:
    """Kafka producer for real-time feature updates"""
    
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS):
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python not installed")
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3
        )
    
    def publish_feature_update(self, entity_id: str, feature_data: dict, timestamp: str = None):
        """Publish feature update to Kafka topic"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        message = {
            'entity_id': entity_id,
            'feature_data': feature_data,
            'timestamp': timestamp,
            'event_type': 'feature_update'
        }
        
        try:
            future = self.producer.send(
                FEATURE_UPDATES_TOPIC,
                key=entity_id,
                value=message
            )
            record_metadata = future.get(timeout=10)
            print(f"Published feature update for {entity_id} to partition {record_metadata.partition} at offset {record_metadata.offset}")
            return True
        except KafkaError as e:
            print(f"Failed to publish feature update: {e}")
            return False
    
    def publish_prediction(self, entity_id: str, prediction: float, features: dict, timestamp: str = None):
        """Publish model prediction to Kafka topic"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        message = {
            'entity_id': entity_id,
            'prediction': prediction,
            'features': features,
            'timestamp': timestamp,
            'event_type': 'model_prediction'
        }
        
        try:
            future = self.producer.send(
                MODEL_PREDICTIONS_TOPIC,
                key=entity_id,
                value=message
            )
            record_metadata = future.get(timeout=10)
            print(f"Published prediction for {entity_id} to partition {record_metadata.partition} at offset {record_metadata.offset}")
            return True
        except KafkaError as e:
            print(f"Failed to publish prediction: {e}")
            return False
    
    def close(self):
        """Close the Kafka producer"""
        self.producer.flush()
        self.producer.close()

class KafkaFeatureConsumer:
    """Kafka consumer for real-time feature updates"""
    
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, group_id='ml-feature-consumer'):
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python not installed")
        
        self.consumer = KafkaConsumer(
            FEATURE_UPDATES_TOPIC,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        self.feature_store = SimpleFeatureStore()
        self.running = False
    
    def consume_feature_updates(self, timeout_ms=1000):
        """Consume and process feature updates"""
        print("Starting feature update consumer...")
        self.running = True
        
        try:
            while self.running:
                message_pack = self.consumer.poll(timeout_ms=timeout_ms)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        self._process_feature_message(message)
                        
        except KeyboardInterrupt:
            print("Stopping consumer...")
        finally:
            self.running = False
    
    def _process_feature_message(self, message):
        """Process individual feature update message"""
        try:
            data = message.value
            entity_id = data['entity_id']
            feature_data = data['feature_data']
            timestamp = data['timestamp']
            
            print(f"Processing feature update for {entity_id} at {timestamp}")
            
            # Update feature store with new data
            entity_ids = [entity_id]
            feature_descriptions = {name: f"Real-time updated feature: {name}" 
                                 for name in feature_data.keys()}
            
            self.feature_store.store_features(
                entity_ids, 
                feature_data, 
                feature_descriptions
            )
            
            print(f"Updated feature store for {entity_id} with {len(feature_data)} features")
            
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def stop(self):
        """Stop the consumer"""
        self.running = False
        self.consumer.close()

# Initialize Kafka components (if available)
kafka_producer = None
kafka_consumer = None

if KAFKA_AVAILABLE:
    try:
        kafka_producer = KafkaFeatureProducer()
        print("Kafka producer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Kafka producer: {e}")
        print("Make sure Kafka is running on localhost:9092")


class SparkFeatureEngineer:
    """Large-scale feature engineering using Spark"""
    
    def __init__(self, spark_session):
        self.spark = spark_session
        
    def create_spark_dataframe_from_sklearn_data(self, X, y, feature_names):
        """Convert sklearn dataset to Spark DataFrame"""
        # Create pandas DataFrame first with clean column names
        clean_names = [name.replace(' (cm)', '').replace(' ', '_') for name in feature_names]
        pandas_df = pd.DataFrame(X, columns=clean_names)
        pandas_df['target'] = y
        
        # Convert to Spark DataFrame
        spark_df = self.spark.createDataFrame(pandas_df)
        return spark_df
    
    def engineer_features(self, spark_df):
        """Perform large-scale feature engineering"""
        # Feature 1: Interaction terms
        spark_df = spark_df.withColumn("sepal_area", col("sepal_length") * col("sepal_width"))
        spark_df = spark_df.withColumn("petal_area", col("petal_length") * col("petal_width"))
        
        # Feature 2: Ratios
        spark_df = spark_df.withColumn("sepal_length_to_width_ratio", 
                                      col("sepal_length") / col("sepal_width"))
        spark_df = spark_df.withColumn("petal_length_to_width_ratio", 
                                      col("petal_length") / col("petal_width"))
        
        # Feature 3: Log transformations (handle potential division by zero)
        spark_df = spark_df.withColumn("log_sepal_length", log(col("sepal_length") + 1))
        spark_df = spark_df.withColumn("log_petal_length", log(col("petal_length") + 1))
        
        # Feature 4: Polynomial features (squared terms)
        spark_df = spark_df.withColumn("sepal_length_squared", col("sepal_length") ** 2)
        spark_df = spark_df.withColumn("petal_length_squared", col("petal_length") ** 2)
        
        # Feature 5: Statistical aggregations across features
        spark_df = spark_df.withColumn("feature_sum", 
                                      col("sepal_length") + col("sepal_width") + 
                                      col("petal_length") + col("petal_width"))
        spark_df = spark_df.withColumn("feature_mean", 
                                      col("feature_sum") / 4)
        
        return spark_df
    
    def create_feature_pipeline(self, feature_columns):
        """Create Spark ML pipeline for feature transformation"""
        # Assemble features into vector
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features_vector"
        )
        
        # Standardize features
        scaler = StandardScaler(
            inputCol="features_vector",
            outputCol="scaled_features",
            withMean=True,
            withStd=True
        )
        
        # PCA for dimensionality reduction (optional)
        import builtins
        k_value = builtins.min(10, len(feature_columns)) if len(feature_columns) > 10 else len(feature_columns)
        pca = PCA(
            inputCol="scaled_features",
            outputCol="pca_features",
            k=k_value
        )
        
        # Create pipeline
        pipeline = SparkPipeline(stages=[assembler, scaler, pca])
        
        return pipeline

# Load and prepare data
X, y = datasets.load_iris(return_X_y=True)
iris_features_names = datasets.load_iris().feature_names

# Create initial DataFrame for validation
initial_df = pd.DataFrame(X, columns=iris_features_names)
initial_df['target'] = y

# Stanford-inspired data quality validation
print("=== Data Quality Validation ===")
validation_results = data_validator.validate_dataset(initial_df, "iris_initial_dataset")
print(f"Dataset validated: {validation_results['row_count']} rows, {validation_results['column_count']} columns")
print(f"Null counts: {validation_results['null_counts']}")
print(f"Memory usage: {validation_results['memory_usage']} bytes")

# Collect system metrics before processing
initial_metrics = performance_monitor.collect_system_metrics()
print(f"Initial system metrics: CPU {initial_metrics['cpu_percent']}%, Memory {initial_metrics['memory_percent']}%")

# Network analysis on features
print("\n=== Feature Network Analysis ===")
feature_network = network_analyzer.build_feature_correlation_network(initial_df, threshold=0.3)
network_metrics = network_analyzer.get_network_metrics()
print(f"Feature network: {network_metrics['node_count']} nodes, {network_metrics['edge_count']} edges")
print(f"Network density: {network_metrics['density']:.3f}")

feature_engineer = SparkFeatureEngineer(spark)
spark_df = feature_engineer.create_spark_dataframe_from_sklearn_data(X, y, iris_features_names)

print("Original Spark DataFrame schema:")
spark_df.printSchema()

print(f"Original data shape: {spark_df.count()} rows")

# Perform large-scale feature engineering
engineered_spark_df = feature_engineer.engineer_features(spark_df)

print("\nEngineered Spark DataFrame schema:")
engineered_spark_df.printSchema()
print(f"Engineered data shape: {engineered_spark_df.count()} rows")

# Convert to pandas for additional processing
engineered_pandas_df = engineered_spark_df.toPandas()

# Validate engineered data
engineered_validation = data_validator.validate_dataset(engineered_pandas_df, "iris_engineered_dataset")
print(f"\nEngineered data validation: {engineered_validation['row_count']} rows, {engineered_validation['column_count']} columns")

# Check for data drift
drift_results = data_validator.check_data_drift(engineered_pandas_df, initial_df)
print(f"Data drift detected in {sum(1 for r in drift_results.values() if r['drift_detected'])} features")

# Use distributed processing for large datasets
print("\n=== Distributed Processing ===")
cluster_info = distributed_processor.get_cluster_info()
if "error" not in cluster_info:
    print(f"Dask cluster: {cluster_info['workers']} workers, {cluster_info['cores_available']} cores")
    processed_df = distributed_processor.process_large_dataset(engineered_pandas_df, "feature_engineering")
    print(f"Distributed processing completed: {len(processed_df)} rows")
else:
    print(cluster_info["error"])

feature_columns = [col for col in engineered_spark_df.columns if col != "target"]

feature_pipeline = feature_engineer.create_feature_pipeline(feature_columns)
pipeline_model = feature_pipeline.fit(engineered_spark_df)

# Transform data
transformed_spark_df = pipeline_model.transform(engineered_spark_df)

print("\nTransformed Spark DataFrame with pipeline features:")
transformed_spark_df.select("target", "features_vector", "scaled_features", "pca_features").show(5)

# Split the transformed data
transformed_pandas_df = transformed_spark_df.toPandas()
# Drop Spark ML vector columns before sklearn training
transformed_pandas_df = transformed_pandas_df.drop(['features_vector', 'scaled_features', 'pca_features'], axis=1, errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(
    transformed_pandas_df.drop('target', axis=1), 
    transformed_pandas_df['target'], 
    test_size=0.2, 
    random_state=42
)

# Store training features in feature store
train_entity_ids = [f"sample_{i}" for i in range(len(X_train))]
# Store all engineered features (exclude Spark ML vector columns)
train_feature_data = {}
for column in X_train.columns:
    # Skip Spark ML vector columns that can't be stored in SQLite
    if column in ['features_vector', 'scaled_features', 'pca_features']:
        continue
    # Convert numpy arrays to lists for SQLite compatibility
    if hasattr(X_train[column].iloc[0], 'tolist'):
        train_feature_data[column] = X_train[column].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x).tolist()
    else:
        train_feature_data[column] = X_train[column].tolist()
    
# Add target
train_feature_data["target"] = y_train.tolist()

# Feature descriptions for engineered features
engineered_feature_descriptions = {
    "sepal_area": "Area of sepal (length * width)",
    "petal_area": "Area of petal (length * width)",
    "sepal_length_to_width_ratio": "Ratio of sepal length to width",
    "petal_length_to_width_ratio": "Ratio of petal length to width",
    "log_sepal_length": "Log transformed sepal length",
    "log_petal_length": "Log transformed petal length",
    "sepal_length_squared": "Squared sepal length",
    "petal_length_squared": "Squared petal length",
    "feature_sum": "Sum of all four original features",
    "feature_mean": "Mean of all four original features"
}
 
# Add original feature descriptions
original_feature_descriptions = {
    "sepal_length": "Length of sepal in cm",
    "sepal_width": "Width of sepal in cm", 
    "petal_length": "Length of petal in cm",
    "petal_width": "Width of petal cm"
}

# Combine all feature descriptions
all_feature_descriptions = {**engineered_feature_descriptions, **original_feature_descriptions}

# Store features
feature_store.store_features(train_entity_ids, train_feature_data, all_feature_descriptions)

# Publish training features to Kafka if available
if kafka_producer:
    for i, entity_id in enumerate(train_entity_ids):
        feature_update = {}
        for feature_name in train_feature_data.keys():
            if feature_name != "target" and i < len(train_feature_data[feature_name]):
                feature_update[feature_name] = train_feature_data[feature_name][i]
        
        if feature_update:
            kafka_producer.publish_feature_update(entity_id, feature_update)

print(f"Stored {len(train_entity_ids)} training samples in feature store")
print(f"Total features per sample: {len(train_feature_data) - 1}")  # exclude target
print("Available engineered features:", [col for col in X_train.columns if col not in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 42
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# Store test features in feature store
test_entity_ids = [f"test_sample_{i}" for i in range(len(X_test))]
test_feature_data = {}
for column in X_test.columns:
    # Skip Spark ML vector columns that can't be stored in SQLite
    if column in ['features_vector', 'scaled_features', 'pca_features']:
        continue
    # Convert numpy arrays to lists for SQLite compatibility
    if hasattr(X_test[column].iloc[0], 'tolist'):
        test_feature_data[column] = X_test[column].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x).tolist()
    else:
        test_feature_data[column] = X_test[column].tolist()
test_feature_data["predicted_target"] = y_pred.tolist()


feature_store.store_features(test_entity_ids, test_feature_data)

# Publish test features and predictions to Kafka if available
if kafka_producer:
    for i, entity_id in enumerate(test_entity_ids):
        feature_update = {}
        for feature_name in test_feature_data.keys():
            if feature_name != "predicted_target" and i < len(test_feature_data[feature_name]):
                feature_update[feature_name] = test_feature_data[feature_name][i]
        
        if feature_update:
            kafka_producer.publish_feature_update(entity_id, feature_update)
            
        # Publish prediction
        if i < len(y_pred):
            kafka_producer.publish_prediction(entity_id, y_pred[i], feature_update)

# Retrieve features from feature store for validation
retrieved_features = feature_store.get_features(
    test_entity_ids[:5], 
    ["sepal_length", "sepal_width", "petal_length", "petal_width"]
)
print("\nRetrieved features from feature store:")
print(retrieved_features.head())

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)





mlflow.set_experiment("iris_logistic_regression_with_spark_features")
with mlflow.start_run():
    mlflow.log_params(params)
    
    # Log basic metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r2": r2,
        "mse": mse
    })
    
    # Log feature engineering metrics
    mlflow.log_metrics({
        "num_training_features": len(train_entity_ids),
        "num_test_features": len(test_entity_ids),
        "num_original_features": 4,
        "num_engineered_features": len([col for col in X_train.columns if col not in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]),
        "total_features": len(train_feature_data) - 1,  # exclude target
        "feature_store_db_size": os.path.getsize("feature_store.db") if os.path.exists("feature_store.db") else 0
    })
    
    # Log feature metadata as artifact
    feature_metadata = feature_store.get_feature_metadata()
    feature_metadata.to_csv("feature_metadata.csv", index=False)
    mlflow.log_artifact("feature_metadata.csv")
    
    mlflow.set_tag("Training info", "logistic_regression with Spark-engineered features")
    mlflow.set_tag("Feature Engineering", "Apache Spark large-scale feature engineering")
    mlflow.set_tag("Feature Store", "Simple SQLite-based feature store")
    
    signature = models.infer_signature(X_train, y_train)
    
    model_info = mlflow.sklearn.log_model(
        sk_model = lr,
        artifact_path = "iris_model_spark_features",
        signature = signature,
        input_example = X_train,
        registered_model_name = "iris_with_spark_features"
    )
    
    

    
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)


# Create results DataFrame
results = pd.DataFrame(X_test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
results["predictions"] = predictions
results["actual"] = y_test
results["difference"] = results["predictions"] - results["actual"]

# Demonstrate real-time feature updates
print("\n=== Real-time Feature Update Demonstration ===")
if kafka_producer:
    # Simulate real-time feature updates for a few samples
    sample_updates = [
        {"entity_id": "real_time_sample_1", "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"entity_id": "real_time_sample_2", "sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3},
        {"entity_id": "real_time_sample_3", "sepal_length": 7.3, "sepal_width": 2.8, "petal_length": 6.1, "petal_width": 1.8}
    ]
    
    for update in sample_updates:
        entity_id = update.pop("entity_id")
        kafka_producer.publish_feature_update(entity_id, update)
        
        # Make prediction on new features
        import numpy as np
        feature_array = np.array([[update["sepal_length"], update["sepal_width"], 
                                   update["petal_length"], update["petal_width"]]])
        
        # Need to engineer the same features for prediction
        feature_df = pd.DataFrame(feature_array, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        feature_df['sepal_area'] = feature_df['sepal_length'] * feature_df['sepal_width']
        feature_df['petal_area'] = feature_df['petal_length'] * feature_df['petal_width']
        feature_df['sepal_length_to_width_ratio'] = feature_df['sepal_length'] / feature_df['sepal_width']
        feature_df['petal_length_to_width_ratio'] = feature_df['petal_length'] / feature_df['petal_width']
        feature_df['log_sepal_length'] = np.log(feature_df['sepal_length'] + 1)
        feature_df['log_petal_length'] = np.log(feature_df['petal_length'] + 1)
        feature_df['sepal_length_squared'] = feature_df['sepal_length'] ** 2
        feature_df['petal_length_squared'] = feature_df['petal_length'] ** 2
        feature_df['feature_sum'] = (feature_df['sepal_length'] + feature_df['sepal_width'] + 
                                   feature_df['petal_length'] + feature_df['petal_width'])
        feature_df['feature_mean'] = feature_df['feature_sum'] / 4
        
        # Align columns with training data
        for col in X_train.columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        feature_df = feature_df[X_train.columns]  # Ensure same order
        prediction = lr.predict(feature_df)[0]
        
        kafka_producer.publish_prediction(entity_id, prediction, update)
        
        print(f"Real-time update for {entity_id}: Features={update}, Prediction={prediction}")
        
        time.sleep(1)  # Simulate real-time delay
    
    print("\nTo consume real-time updates, run:")
    print("consumer = KafkaFeatureConsumer()")
    print("# Run in separate thread or process")
    print("consumer.consume_feature_updates()")
else:
    print("Kafka not available - skipping real-time demonstration")

# Close Kafka producer
if kafka_producer:
    kafka_producer.close()

# Stop Spark session
spark.stop()

print(results)

