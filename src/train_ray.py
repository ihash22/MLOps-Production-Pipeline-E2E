import ray
from ray import tune
from ray.tune import RunConfig, TuneConfig
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

# Define the absolute path to your project's mlruns folder
BASE_DIR = Path(__file__).resolve().parent.parent
MLRUNS_DIR = f"file://{BASE_DIR}/mlruns"

def train_iris(config):
    # Explicitly tell this isolated Ray worker where to log data
    mlflow.set_tracking_uri(MLRUNS_DIR)
    mlflow.set_experiment("ray_distributed_search")

    # 1. Load Data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # 2. Start MLflow run
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"]
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        # Log metrics to MLflow
        mlflow.log_params(config)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save the physical model artifact
        mlflow.sklearn.log_model(model, "model")
        
        # Report back to Ray
        return {"accuracy": accuracy}

def run_hyperparam_search():
    ray.init(ignore_reinit_error=True)
    
    search_space = {
        "n_estimators": tune.grid_search([10, 50, 100]),
        "max_depth": tune.choice([3, 5, 10])
    }

    print("🚀 Starting Distributed Ray Tune Search...")
    
    tuner = tune.Tuner(
        train_iris,
        param_space=search_space,
        tune_config=TuneConfig(
            metric="accuracy",
            mode="max"
        ),
        run_config=RunConfig(name="ray_iris_search")
    )
    
    results = tuner.fit()
    print("✅ Best hyperparameters found:", results.get_best_result().config)

if __name__ == "__main__":
    run_hyperparam_search()