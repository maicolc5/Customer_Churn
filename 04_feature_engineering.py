"""
04 - FEATURE ENGINEERING
Create new features to improve model performance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

def load_data():
    """Load preprocessed data."""
    df = pd.read_csv(DATA_DIR / "processed_data.csv")
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

def create_recency_features(df):
    """Create features based on purchase recency."""
    print("\nCreating recency features...")
    
    # Recency segments
    df['recency_segment'] = pd.cut(
        df['days_since_last_purchase'],
        bins=[0, 30, 90, 180, 365, np.inf],
        labels=['very_recent', 'recent', 'moderate', 'old', 'very_old']
    )
    
    # Is recent buyer (within 30 days)
    df['is_recent_buyer'] = (df['days_since_last_purchase'] <= 30).astype(int)
    
    # Is at risk (90-180 days)
    df['is_at_risk'] = ((df['days_since_last_purchase'] > 90) & 
                        (df['days_since_last_purchase'] <= 180)).astype(int)
    
    # Is churned (>180 days)
    df['is_churned'] = (df['days_since_last_purchase'] > 180).astype(int)
    
    print("  Created: recency_segment, is_recent_buyer, is_at_risk, is_churned")
    return df

def create_monetary_features(df):
    """Create features based on spending patterns."""
    print("\nCreating monetary features...")
    
    # Spending tier
    df['spending_tier'] = pd.qcut(
        df['total_spent'],
        q=4,
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # Value per product
    df['value_per_product'] = df['total_spent'] / df['unique_products'].replace(0, 1)
    
    # Spending intensity (spent per day of relationship)
    df['spending_intensity'] = df['total_spent'] / df['days_since_first_purchase'].replace(0, 1)
    
    print("  Created: spending_tier, value_per_product, spending_intensity")
    return df

def create_frequency_features(df):
    """Create features based on purchase frequency."""
    print("\nCreating frequency features...")
    
    # Is frequent buyer (>1 order per month)
    df['is_frequent_buyer'] = (df['orders_per_month'] > 1).astype(int)
    
    # Order frequency score (normalized)
    df['frequency_score'] = df['orders_per_month'] / df['orders_per_month'].max()
    
    # Consistency score (lower variance = more consistent)
    df['consistency_score'] = 1 / (1 + df['avg_days_between_orders'])
    
    print("  Created: is_frequent_buyer, frequency_score, consistency_score")
    return df

def create_engagement_features(df):
    """Create features measuring customer engagement."""
    print("\nCreating engagement features...")
    
    # Customer lifetime (days)
    df['customer_lifetime'] = df['days_since_first_purchase']
    
    # Engagement score (combination of frequency and recency)
    df['engagement_score'] = (
        df['frequency_score'] * 0.4 +
        (1 / (1 + df['days_since_last_purchase'] / 365)) * 0.3 +
        df['consistency_score'] * 0.3
    )
    
    # Product diversity
    df['product_diversity'] = df['unique_products'] / df['total_orders'].replace(0, 1)
    
    print("  Created: customer_lifetime, engagement_score, product_diversity")
    return df

def create_interaction_features(df):
    """Create interaction features between existing ones."""
    print("\nCreating interaction features...")
    
    # Recency x Monetary interaction
    df['recency_monetary'] = df['days_since_last_purchase'] * df['total_spent']
    
    # Frequency x Monetary interaction
    df['frequency_monetary'] = df['orders_per_month'] * df['total_spent']
    
    print("  Created: recency_monetary, frequency_monetary")
    return df

def encode_new_categorical(df):
    """Encode newly created categorical features."""
    print("\nEncoding new categorical features...")
    
    categorical_cols = ['recency_segment', 'spending_tier']
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )
    
    print(f"Encoded {len(categorical_cols)} new categorical columns")
    return df_encoded

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Create features
    df = create_recency_features(df)
    df = create_monetary_features(df)
    df = create_frequency_features(df)
    df = create_engagement_features(df)
    df = create_interaction_features(df)
    
    # Encode new categoricals
    df = encode_new_categorical(df)

    # Convert all boolean features to numeric 0/1 values
    boolean_cols = df.select_dtypes(include='bool').columns
    df[boolean_cols] = df[boolean_cols].astype(int)
    
    # Save engineered features
    output_path = DATA_DIR / "engineered_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved engineered data to {output_path}")
    
    # Update feature list
    feature_cols = [c for c in df.columns if c not in ['CustomerID', 'AccountNumber', 'will_buy_again']]
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")
    print(f"Updated feature list: {len(feature_cols)} features")
    
    # Print new features summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Original features: 11")
    print(f"New features created: {len(df.columns) - 11}")
    print(f"Total features: {len(df.columns) - 3}")  # Excluding IDs and target
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print("Next step: Run 05_model_selection.py")

if __name__ == "__main__":
    main()
