"""
03 - DATA PREPROCESSING
Clean and prepare data for modeling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load raw data."""
    df = pd.read_csv(DATA_DIR / "raw_customer_features.csv")
    print(f"Loaded {len(df)} rows")
    return df

def handle_missing_values(df):
    """Handle missing values."""
    print("\nHandling missing values...")

    # Clientes sin historial: los valores de compra representan cero
    no_history = df['total_orders'].eq(0)

    zero_cols = [
        'total_spent',
        'avg_order_value',
        'days_since_last_purchase',
        'days_since_first_purchase'
    ]

    df.loc[no_history, zero_cols] = 0
    
    # Check missing
    missing = df.isnull().sum()
    if missing.any():
        print(f"Missing values found:\n{missing[missing > 0]}")
        
        # Numeric: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  Filled {col} with median: {median_val:.2f}")
                
        
        # Keep missing categorical information explicit instead of assigning
        # an arbitrary category such as the global mode.
        cat_cols = df.select_dtypes(include=['object', 'str']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
                print(f"  Filled {col} with category: Unknown")
    else:
        print("No missing values found.")

    # Preserve whether an interval between repeat orders can be measured.
    df['has_repeat_order_history'] = (df['total_orders'] > 1).astype(int)

    repeat_order_median = df.loc[
        df['total_orders'] > 1,
        'avg_days_between_orders'
    ].median()
    df['avg_days_between_orders'] = df[
        'avg_days_between_orders'
    ].replace(9999, repeat_order_median)
    print(f"  Replaced 9999-day sentinel with median: {repeat_order_median:.2f}")
    
    return df

def remove_duplicates(df):
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicates")
    return df

def handle_outliers(df, numeric_cols, method='iqr', threshold=1.5):
    """Handle outliers using IQR method."""
    print("\nHandling outliers...")
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            # Cap outliers instead of removing
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"  {col}: Capped {outliers} outliers")
    
    return df

def encode_categorical(df, categorical_cols):
    """Encode categorical variables."""
    print("\nEncoding categorical variables...")
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"Encoded {len(categorical_cols)} categorical columns")
    print(f"New shape: {df_encoded.shape}")
    
    return df_encoded

def scale_features(df, numeric_cols):
    """Scale numeric features."""
    from sklearn.preprocessing import StandardScaler
    
    print("\nScaling features...")
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Saved scaler to {MODELS_DIR / 'scaler.pkl'}")
    
    return df

def main():
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Load data
    df = load_data()

    # Create a new feature indicating if the customer has any purchase history, 
    # to identify customers with no prior purchases. This can be useful for modeling and segmentation.
    df['has_purchase_history'] = (df['total_orders'] > 0).astype(int)
    
    # Store original columns for reference
    original_cols = df.columns.tolist()
    
    # 1. Remove duplicates
    df = remove_duplicates(df)
    
    # 2. Handle missing values
    df = handle_missing_values(df)

    # Group cities representing less than 0.5% of the dataset.
    city_share = df['city'].value_counts(normalize=True)
    rare_cities = city_share[city_share < 0.005].index
    df['city'] = df['city'].where(
        ~df['city'].isin(rare_cities),
        'Other'
    )
    print(f"Grouped {len(rare_cities)} rare cities into Other")
    
    # 3. Identify column types
    numeric_cols = [
        'total_orders',
        'total_spent',
        'avg_order_value',
        'unique_products',
        'orders_per_month'
    ]
    
    categorical_cols = ['favorite_category', 'city']
    
    # 4. Handle outliers
    df = handle_outliers(df, numeric_cols)
    
    # 5. Encode categorical variables
    df_encoded = encode_categorical(df, categorical_cols)
    
    # 6. Scale features (optional - some models don't need it)
    # df_scaled = scale_features(df_encoded, numeric_cols)
    
    # Save processed data
    output_path = DATA_DIR / "processed_data.csv"
    df_encoded.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")
    
    # Save feature list
    feature_cols = [c for c in df_encoded.columns if c not in [
        'CustomerID', 'AccountNumber', 'will_buy_soon',
        'will_buy_again_6m', 'customer_cohort'
    ]]
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")
    print(f"Saved {len(feature_cols)} feature columns")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("Next step: Run 04_feature_engineering.py")

if __name__ == "__main__":
    main()
