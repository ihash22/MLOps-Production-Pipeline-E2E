import mlflow
import os
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- ROBUST PATHING ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MLRUNS_DIR = BASE_DIR / "mlruns"
os.makedirs(MLRUNS_DIR, exist_ok=True)

# Use the absolute local filesystem URI
TRACKING_URI = f"file://{MLRUNS_DIR}"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("iris_production_check")

def train_and_log(C_value=1.0):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"C_{C_value}"):
        # Log parameters
        mlflow.log_param("C", C_value)
        mlflow.log_param("model_type", "LogisticRegression")
        
        # Train
        model = LogisticRegression(C=C_value, max_iter=200)
        model.fit(X_train, y_train)
        
        # Log metrics
        acc = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        
        # Log the actual model artifact (Critical for Week 2/3)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Target Reached: C={C_value}, Accuracy={acc:.4f}")

if __name__ == "__main__":
    print(f"Tracking to: {TRACKING_URI}")
    # We manually run 3 variations to simulate a sweep without the Ray overhead
    for c in [0.01, 0.1, 1.0]:
        train_and_log(c)
    print("🚀 All runs completed successfully!")