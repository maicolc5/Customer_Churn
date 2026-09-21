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
            # Keep missing population values available for an explicit Unknown
            # city-size category instead of classifying them as median-sized.
            if col == 'population':
                continue
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


def build_major_city(df):
    """Classify cities independently within each country.

    The top 25% of cities by population within each country are Major City;
    the remaining cities are Small City. Missing geography stays Unknown.
    """
    print("\nClassifying cities by country...")

    if 'city' not in df.columns:
        print("No city column found; skipping geospatial grouping.")
        return df

    df['city'] = df['city'].fillna('Unknown').astype(str).str.strip().str.title()
    df['city'] = df['city'].replace(['Unknown', 'Uknow', 'N/A', ''], 'Other')

    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown').astype(str).str.strip().str.title()

    # Classify one population value per city so customer volume does not
    # distort the country-relative ranking.
    df['city_class'] = 'Unknown'
    if 'country' in df.columns and 'population' in df.columns:
        city_reference = (
            df.loc[
                (df['country'] != 'Unknown') &
                (df['city'] != 'Other') &
                df['population'].notna(),
                ['country', 'city', 'population']
            ]
            .groupby(['country', 'city'], as_index=False)['population']
            .median()
        )
        city_reference['country_rank'] = city_reference.groupby('country')['population'].rank(
            pct=True,
            method='average'
        )
        city_reference['city_class'] = np.where(
            city_reference['country_rank'] >= 0.75,
            'Major City',
            'Small City'
        )
        class_map = city_reference.set_index(['country', 'city'])['city_class']
        df['city_class'] = [
            class_map.get((country, city), 'Unknown')
            for country, city in zip(df['country'], df['city'])
        ]

    df['city_class'] = df['city_class'].fillna('Unknown')
    if 'population' in df.columns and df['population'].isna().any():
        df['population'] = df['population'].fillna(df['population'].median())
    print(f"  Final city classes: {df['city_class'].nunique()}")
    print("  City-class distribution by country:")
    print(df.groupby(['country', 'city_class'], dropna=False).size().to_string())
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

    # 3. Classify cities relative to other cities in the same country.
    df = build_major_city(df)

    # 4. Identify column types
    numeric_cols = [
        'total_orders',
        'total_spent',
        'avg_order_value',
        'unique_products',
        'orders_per_month'
    ]
    
    # Keep raw geography until feature engineering calculates nearest major city.
    df = df.drop(columns=['postal_code'], errors='ignore')
    categorical_cols = ['favorite_category']
    
    # 5. Handle outliers
    df = handle_outliers(df, numeric_cols)
    
    # 6. Encode categorical variables
    df_encoded = encode_categorical(df, categorical_cols)

    # 7. Convert boolean columns to 0/1 before saving the final dataset.
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
    
    # 8. Scale features (optional - some models don't need it)
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
