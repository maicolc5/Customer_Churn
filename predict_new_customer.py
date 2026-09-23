"""
Predict six-month repurchase probability for a customer.
"""
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"


def load_artifacts():
    """Load the trained model and the exact training feature order."""
    model = joblib.load(MODELS_DIR / "trained_model.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    return model, feature_columns


def predict_customer(customer_features):
    """Return six-month repurchase probability for one feature dictionary."""
    model, feature_columns = load_artifacts()
    customer_df = pd.DataFrame([customer_features])

    missing_columns = [
        column for column in feature_columns
        if column not in customer_df.columns
    ]
    extra_columns = [
        column for column in customer_df.columns
        if column not in feature_columns
    ]
    if missing_columns or extra_columns:
        raise ValueError(
            f"Missing features: {missing_columns}; "
            f"Unexpected features: {extra_columns}"
        )

    customer_df = customer_df[feature_columns]
    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = int(probability >= 0.5)

    return {
        "will_buy_again_6m": prediction,
        "probability_return_6m": probability,
        "probability_no_return_6m": 1 - probability,
    }


def predict_csv(input_path, output_path=None):
    """Predict all rows in a CSV containing the 43 engineered features."""
    model, feature_columns = load_artifacts()
    customer_df = pd.read_csv(input_path)

    missing_columns = [
        column for column in feature_columns
        if column not in customer_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing features: {missing_columns}")

    probabilities = model.predict_proba(customer_df[feature_columns])[:, 1]
    customer_df["probability_return_6m"] = probabilities
    customer_df["probability_no_return_6m"] = 1 - probabilities
    customer_df["will_buy_again_6m"] = (probabilities >= 0.5).astype(int)

    if output_path is None:
        output_path = Path(input_path).with_name("customer_predictions.csv")
    customer_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    print("Usage:")
    print("  from predict_new_customer import predict_customer")
    print("  result = predict_customer({ ...43 engineered features... })")
    print("  py predict_new_customer.py input_features.csv")
