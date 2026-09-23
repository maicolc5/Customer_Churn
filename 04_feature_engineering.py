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


def create_geographic_features(df):
    """Assign each small city to its nearest major city in the same country."""
    print("\nCreating geographic area features...")

    required_columns = {
        'country', 'city', 'city_class', 'latitude', 'longitude'
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing geographic columns: {sorted(missing_columns)}")

    df['geographic_area'] = 'Unknown'
    df['distance_to_major_city_km'] = np.nan

    valid_rows = df[
        df['country'].notna() &
        df['country'].ne('Unknown') &
        df['city'].notna() &
        df['city'].ne('Unknown') &
        df['city_class'].isin(['Major City', 'Small City']) &
        df['latitude'].notna() &
        df['longitude'].notna()
    ]
    major_reference = (
        valid_rows[valid_rows['city_class'].eq('Major City')]
        .groupby(['country', 'city'], as_index=False)
        .agg(
            major_latitude=('latitude', 'mean'),
            major_longitude=('longitude', 'mean')
        )
    )

    earth_radius_km = 6371.0
    for country, country_rows in valid_rows.groupby('country'):
        country_majors = major_reference[
            major_reference['country'].eq(country)
        ]
        if country_majors.empty:
            continue

        row_indexes = country_rows.index.to_numpy()
        latitudes = np.radians(country_rows['latitude'].to_numpy())[:, None]
        longitudes = np.radians(country_rows['longitude'].to_numpy())[:, None]
        major_latitudes = np.radians(
            country_majors['major_latitude'].to_numpy()
        )[None, :]
        major_longitudes = np.radians(
            country_majors['major_longitude'].to_numpy()
        )[None, :]

        delta_latitude = major_latitudes - latitudes
        delta_longitude = major_longitudes - longitudes
        haversine_a = (
            np.sin(delta_latitude / 2) ** 2 +
            np.cos(latitudes) * np.cos(major_latitudes) *
            np.sin(delta_longitude / 2) ** 2
        )
        distances = 2 * earth_radius_km * np.arcsin(
            np.sqrt(np.clip(haversine_a, 0, 1))
        )
        nearest_positions = distances.argmin(axis=1)
        nearest_distances = distances[
            np.arange(len(country_rows)), nearest_positions
        ]
        nearest_cities = country_majors.iloc[nearest_positions]['city'].to_numpy()

        is_major = country_rows['city_class'].eq('Major City').to_numpy()
        area_labels = np.where(
            is_major,
            country_rows['city'].to_numpy(),
            'around_' + nearest_cities
        )
        df.loc[row_indexes, 'geographic_area'] = area_labels
        df.loc[row_indexes, 'distance_to_major_city_km'] = nearest_distances

    df['has_geographic_match'] = df['distance_to_major_city_km'].notna().astype(int)
    df['city_class_major'] = (
        df['city_class'].map({
            'Major City': 1,
            'Small City': 0
        }).fillna(-1).astype(int)
    )
    # Do not replace an unknown geographic distance with a real distance.
    # -1 is outside the valid kilometer range and has_geographic_match marks it.
    df['distance_to_major_city_km'] = df[
        'distance_to_major_city_km'
    ].fillna(-1)
    print(f"  Geographic areas created: {df['geographic_area'].nunique()}")
    print(f"  Small-city assignments: {df['geographic_area'].str.startswith('around_').sum()}")
    return df

def create_recency_features(df):
    """Create recency features using information available at the cutoff."""
    print("\nCreating recency features...")
    
    # Recency segments
    df['recency_segment'] = pd.cut(
        df['days_since_last_purchase'],
        bins=[-np.inf, 30, 90, 180, 365, np.inf],
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
    """Create monetary features from purchases before the cutoff."""
    print("\nCreating monetary features...")
    
    # Spending tier
    spending_codes = pd.qcut(
        df['total_spent'],
        q=4,
        labels=False,
        duplicates='drop'
    )
    spending_labels = ['low', 'medium', 'high', 'very_high']
    df['spending_tier'] = spending_codes.map(
        dict(enumerate(spending_labels))
    ).astype('category')
    
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
    
    # Keep country as the only geographic predictor. Other geographic fields
    # remain available for analysis but are excluded from model features.
    regular_categorical_cols = ['recency_segment', 'spending_tier']
    df_encoded = pd.get_dummies(
        df,
        columns=regular_categorical_cols,
        drop_first=True,
        dtype=int
    )

    country_dummies = pd.get_dummies(
        df_encoded['country'],
        prefix='country',
        drop_first=False,
        dtype=int
    )
    df_encoded = pd.concat(
        [df_encoded.drop(columns=['country']), country_dummies],
        axis=1
    )
    df_encoded = df_encoded.drop(
        columns=['country_Canada', 'country_Unknown'],
        errors='ignore'
    )

    encoded_count = len(regular_categorical_cols) + 1
    print(f"Encoded {encoded_count} categorical columns")
    return df_encoded

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Create features
    df = create_geographic_features(df)
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
    excluded_model_columns = {
        'CustomerID',
        'AccountNumber',
        'will_buy_soon',
        'will_buy_again_6m',
        'customer_cohort',
        'city',
        'country',
        'city_class',
        'geographic_area',
        'latitude',
        'longitude',
        'population',
        'population_missing',
        'distance_to_major_city_km',
        'has_geographic_match',
        'city_class_major'
    }
    feature_cols = [c for c in df.columns if c not in excluded_model_columns]
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
