import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

# --- THE ACTUAL EVIDENTLY 0.7.x IMPORTS ---
from evidently import Report
from evidently.presets import DataDriftPreset

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "docs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run_drift_report():
    # 1. Get Reference Data
    iris = load_iris()
    reference_df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # 2. Simulate "Current" Data with drift
    current_df = reference_df.copy()
    current_df['petal length (cm)'] = current_df['petal length (cm)'] * 1.5 

    # 3. Create the Report (0.7+ syntax: passed as a direct list)
    drift_report = Report([DataDriftPreset()])

    print("📊 Analyzing data for drift...")
    
    # 4. Run the report and capture the RESULT object (0.7+ syntax)
    result = drift_report.run(reference_data=reference_df, current_data=current_df)

    # 5. Save as HTML from the result object (0.7+ syntax)
    report_path = REPORTS_DIR / "data_drift_report.html"
    result.save_html(str(report_path))
    
    print(f"✅ Drift report generated at: {report_path}")

if __name__ == "__main__":
    run_drift_report()