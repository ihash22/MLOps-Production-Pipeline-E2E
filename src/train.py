import mlflow
import os
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- ROBUST PATHING (Ensure these remain at the top of your script) ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MLRUNS_DIR = BASE_DIR / "mlruns"
os.makedirs(MLRUNS_DIR, exist_ok=True)
TRACKING_URI = f"file://{MLRUNS_DIR}"

def train_and_log(C_value=1.0):
    """
    Trains a Logistic Regression model and logs results to MLflow.
    Includes defensive logic to ensure experiment exists in the current tracking URI.
    """
    # 1. Force MLflow to use the correct URI (Crucial for pytest isolation)
    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    
    experiment_name = "iris_production_check"
    
    # 2. Safely get or create the experiment to avoid ID Mismatches
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # 3. Load and Split Data
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # 4. Start Run using the explicit experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"C_{C_value}"):
        # Log parameters
        mlflow.log_param("C", C_value)
        mlflow.log_param("model_type", "LogisticRegression")
        
        # Train model
        model = LogisticRegression(C=C_value, max_iter=200)
        model.fit(X_train, y_train)
        
        # Calculate and log metrics
        acc = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        
        # Log the actual model artifact for future deployment
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Run Successful: C={C_value}, Accuracy={acc:.4f}")

if __name__ == "__main__":
    # Standard production execution
    mlflow.set_tracking_uri(TRACKING_URI)
    for c in [0.01, 0.1, 1.0]:
        train_and_log(c)