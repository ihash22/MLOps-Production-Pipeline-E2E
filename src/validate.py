import great_expectations as gx
import pandas as pd
from sklearn.datasets import load_iris

def run_quick_validation():
    # 1. Get your context
    context = gx.get_context()

    # 2. Load some sample data
    data = load_iris(as_frame=True)
    df = data.frame

    # 3. Create a temporary validator (In v1, this is the easiest 'quick start')
    validator = context.sources.pandas_default.read_dataframe(df, name="iris_data")

    # 4. Add an expectation
    validator.expect_column_to_exist("target")
    validator.expect_column_values_to_not_be_null("target")

    # 5. Run it
    result = validator.validate()
    
    if result["success"]:
        print("🎉 Week 1 Success: Data is valid and GE is working!")
    else:
        print("❌ Validation failed.")

if __name__ == "__main__":
    run_quick_validation()
