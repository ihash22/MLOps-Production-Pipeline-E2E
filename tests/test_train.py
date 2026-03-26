import pytest
import mlflow
import shutil
from pathlib import Path
from src.train import train_and_log

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MLRUNS_DIR = SCRIPT_DIR / "test_mlruns"

@pytest.fixture(scope="function", autouse=True)
def setup_test_env():
    """Ensures a completely isolated MLflow environment for each test."""
    mlflow.end_run() 
    
    # Clean setup
    if TEST_MLRUNS_DIR.exists():
        shutil.rmtree(TEST_MLRUNS_DIR)
    TEST_MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    tracking_uri = f"file://{TEST_MLRUNS_DIR}"
    mlflow.set_tracking_uri(tracking_uri)
    
    yield
    
    # Clean up after test
    mlflow.end_run()

def test_training_pipeline_e2e():
    """
    E2E Test with Debugging: Prints exactly where MLflow is saving data.
    """
    print(f"\n[DEBUG] Expected Tracking URI: file://{TEST_MLRUNS_DIR}")
    
    try:
        train_and_log(C_value=1.0)
        
        # Get the active run to see where it actually went
        active_run = mlflow.last_active_run()
        if active_run:
            print(f"[DEBUG] Actual Run ID: {active_run.info.run_id}")
            print(f"[DEBUG] Actual Artifact URI: {active_run.info.artifact_uri}")
        else:
            print("[DEBUG] No active run found after training!")
            
    except Exception as e:
        pytest.fail(f"Pipeline crashed: {e}")

    # Search for any directory at all in the test folder
    all_files = list(TEST_MLRUNS_DIR.rglob("*"))
    print(f"[DEBUG] Files found in TEST_MLRUNS_DIR: {[f.name for f in all_files[:10]]}")

    # Use a more generic search for any folder containing 'model'
    valid_model_artifacts = list(TEST_MLRUNS_DIR.rglob("model.pkl"))

    assert len(valid_model_artifacts) > 0, (
        f"Model artifact not found in {TEST_MLRUNS_DIR}. "
        f"Check terminal output above for '[DEBUG]' paths."
    )
    
    print(f"✅ Found valid model artifact at: {valid_model_artifacts[0]}")