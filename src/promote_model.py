import mlflow
from mlflow.tracking import MlflowClient
import os
from pathlib import Path

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MLRUNS_DIR = BASE_DIR / "mlruns"
TRACKING_URI = f"file://{MLRUNS_DIR}"
EXPERIMENT_NAME = "iris_production_check"

def get_best_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # 1. Fetch experiment metadata
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
        return

    # 2. Search for all completed runs, sorted by accuracy (descending)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.accuracy > 0",
        order_by=["metrics.accuracy DESC"]
    )

    if not runs:
        print("❌ No successful runs found to promote.")
        return

    best_run = runs[0]
    best_acc = best_run.data.metrics['accuracy']
    best_run_id = best_run.info.run_id

    print(f"🏆 Best Model Found!")
    print(f"Run ID: {best_run_id}")
    print(f"Accuracy: {best_acc:.4f}")

    # 3. Define where to "stage" the model for Week 3
    # We will save the Run ID to a text file so Docker/GitHub can find it
    dist_dir = BASE_DIR / "dist"
    os.makedirs(dist_dir, exist_ok=True)
    
    with open(dist_dir / "best_run.txt", "w") as f:
        f.write(best_run_id)
    
    print(f"✅ Best Run ID saved to {dist_dir}/best_run.txt")

if __name__ == "__main__":
    get_best_model()