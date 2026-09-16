"""
MAIN PIPELINE
Run the entire churn prediction pipeline from start to finish.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("01_data_extraction.py", "Data Extraction"),
    ("02_eda.py", "Exploratory Data Analysis"),
    ("03_preprocessing.py", "Data Preprocessing"),
    ("04_feature_engineering.py", "Feature Engineering"),
    ("05_model_selection.py", "Model Selection"),
    ("06_training.py", "Model Training"),
    ("07_evaluation.py", "Model Evaluation")
]

def run_script(script_name, description):
    """Run a single script."""
    print("\n" + "=" * 60)
    print(f"RUNNING: {description}")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\nError running {script_name}")
        return False
    return True

def main():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 60)
    print("\nThis pipeline will:")
    print("1. Extract data from SQL Server (or generate sample data)")
    print("2. Perform Exploratory Data Analysis")
    print("3. Preprocess and clean the data")
    print("4. Engineer new features")
    print("5. Compare multiple algorithms")
    print("6. Train the best model with hyperparameter tuning")
    print("7. Evaluate and visualize results")
    
    response = input("\nDo you want to run the full pipeline? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelled.")
        return
    
    for script_name, description in SCRIPTS:
        success = run_script(script_name, description)
        if not success:
            print(f"\nPipeline failed at {description}")
            return
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nResults saved in:")
    print("  - data/ (all data files)")
    print("  - models/ (trained model)")
    print("  - *.png (visualizations)")
    print("\nNext: Open 08_conclusion.ipynb to review final analysis")

if __name__ == "__main__":
    main()
