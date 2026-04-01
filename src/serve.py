import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import os

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "dist"
MLRUNS_DIR = BASE_DIR / "mlruns"
TRACKING_URI = f"file://{MLRUNS_DIR}"

app = FastAPI(title="Iris Prediction Service")

def load_best_model():
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    
    try:
        # 1. Read the promoted Run ID from Week 2
        with open(DIST_DIR / "best_run.txt", "r") as f:
            run_id = f.read().strip()
        
        # 2. Find the Experiment folder (e.g., '875516197735633774')
        # We search for the parent of the run_id folder
        run_folders = list(MLRUNS_DIR.rglob(run_id))
        if not run_folders:
            raise FileNotFoundError(f"Run {run_id} not found in {MLRUNS_DIR}")
        
        experiment_root = run_folders[0].parent
        
        # 3. RECURSIVE SEARCH: Find any folder containing 'MLmodel' 
        # inside this specific experiment
        all_mlmodels = list(experiment_root.rglob("MLmodel"))
        
        if not all_mlmodels:
            raise FileNotFoundError(f"No model artifacts found in experiment {experiment_root.name}")

        # MLflow creates multiple model versions; we'll pick the most recent
        # by checking the parent directory of the MLmodel file
        model_path = all_mlmodels[0].parent

        print(f"📦 Successfully matched Run {run_id} to Model at: {model_path}")
        
        # Convert PosixPath to string for MLflow compatibility
        return mlflow.sklearn.load_model(str(model_path))
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise e

model = load_best_model()

# 2. Define the Request Schema
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisRequest):
    # Convert request to DataFrame for the model
    df = pd.DataFrame([data.model_dump().values()], 
                      columns=['sepal length (cm)', 'sepal width (cm)', 
                               'petal length (cm)', 'petal width (cm)'])
    
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

@app.get("/health")
def health():
    return {"status": "healthy"}