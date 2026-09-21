"""
02 - EXPLORATORY DATA ANALYSIS (EDA)
Analyze customer data to understand patterns and relationships.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def load_data():
    """Load the raw customer features."""
    df = pd.read_csv(DATA_DIR / "raw_customer_features.csv")
    return df


def load_engineered_data():
    """Load engineered features for model-specific visualizations."""
    return pd.read_csv(DATA_DIR / "engineered_features.csv")

def basic_info(df):
    """Print basic dataset information."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")

def analyze_cohorts(df):
    """Summarize all customer cohorts before modeling."""
    print("\n" + "=" * 60)
    print("CUSTOMER COHORT ANALYSIS")
    print("=" * 60)

    cohort_counts = df['customer_cohort'].value_counts()
    cohort_pct = df['customer_cohort'].value_counts(normalize=True) * 100

    for cohort, count in cohort_counts.items():
        print(f"  {cohort}: {count} ({cohort_pct[cohort]:.1f}%)")


def analyze_geography(df):
    """Summarize country-relative city classes and geodata coverage."""
    print("\n" + "=" * 60)
    print("GEOGRAPHIC ANALYSIS")
    print("=" * 60)

    if 'country' not in df.columns:
        print("Geographic columns are not available.")
        return

    print("\nCustomers by country:")
    print(df['country'].fillna('Unknown').value_counts().to_string())

    if 'city_class' not in df.columns and {'country', 'city', 'population'}.issubset(df.columns):
        city_reference = (
            df[['country', 'city', 'population']]
            .dropna(subset=['country', 'city', 'population'])
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
        df = df.merge(
            city_reference[['country', 'city', 'city_class']],
            on=['country', 'city'],
            how='left'
        )
        df['city_class'] = df['city_class'].fillna('Unknown')

    if 'city_class' in df.columns:
        print("\nCity classification within each country:")
        city_summary = pd.crosstab(
            df['country'].fillna('Unknown'),
            df['city_class'].fillna('Unknown')
        )
        print(city_summary.to_string())

    geo_columns = [
        column for column in ['latitude', 'longitude', 'population']
        if column in df.columns
    ]
    if geo_columns:
        print("\nGeographic data coverage:")
        coverage = df[geo_columns].notna().mean().mul(100).round(1)
        print(coverage.map(lambda value: f'{value}%').to_string())

    if 'population' in df.columns:
        print("\nPopulation by city class:")
        print(
            df.groupby('city_class', dropna=False)['population']
            .median()
            .sort_values(ascending=False)
            .to_string()
        )


def analyze_geography_by_target(df, target_column):
    """Compare purchase rate across the compact geographic features."""
    print(f"\nGeographic purchase rates for {target_column}:")

    if (
        'city_class' not in df.columns and
        {'country', 'city', 'population'}.issubset(df.columns)
    ):
        city_reference = (
            df[['country', 'city', 'population']]
            .dropna(subset=['country', 'city', 'population'])
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
        df = df.merge(
            city_reference[['country', 'city', 'city_class']],
            on=['country', 'city'],
            how='left'
        )
        df['city_class'] = df['city_class'].fillna('Unknown')

    for column in ['country', 'city_class']:
        if column not in df.columns:
            continue
        purchase_rate = (
            df.assign(**{column: df[column].fillna('Unknown')})
            .groupby(column)[target_column]
            .agg(['count', 'mean'])
        )
        purchase_rate['purchase_rate_pct'] = (purchase_rate['mean'] * 100).round(2)
        print(f"\n{column}:")
        print(
            purchase_rate[['count', 'purchase_rate_pct']]
            .sort_values('purchase_rate_pct', ascending=False)
            .to_string()
        )

    if 'distance_to_major_city_km' in df.columns:
        unknown_distance_count = df['distance_to_major_city_km'].lt(0).sum()
        if unknown_distance_count:
            print(f"  Unknown distance records: {unknown_distance_count}")
        valid_distance_df = df[
            df['distance_to_major_city_km'].ge(0)
        ].copy()
        distance_bins = pd.cut(
            valid_distance_df['distance_to_major_city_km'],
            bins=[-np.inf, 50, 150, 300, np.inf],
            labels=['0-50 km', '50-150 km', '150-300 km', '300+ km']
        )
        distance_rates = valid_distance_df.groupby(
            distance_bins, observed=False
        )[target_column].agg(
            count='size',
            purchase_rate='mean'
        )
        distance_rates['purchase_rate_pct'] = (
            distance_rates['purchase_rate'] * 100
        ).round(2)
        print("\nDistance to major city:")
        print(distance_rates[['count', 'purchase_rate_pct']].to_string())


def plot_geographic_distributions(df):
    """Plot population and distance distributions for model customers."""
    figures = [
        ('population', 'Population distribution (log scale)', 'population_distribution.png'),
        (
            'distance_to_major_city_km',
            'Distance to major city distribution',
            'distance_to_major_city_distribution.png'
        )
    ]

    for column, title, filename in figures:
        if column not in df.columns:
            continue
        values = df[column].dropna()
        if column == 'population':
            values = np.log1p(values)
            xlabel = 'log(1 + population)'
        else:
            values = values[values.ge(0)]
            xlabel = 'Distance (km)'

        plt.figure(figsize=(9, 5))
        plt.hist(values, bins=30, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Customers')
        plt.tight_layout()
        output_path = DATA_DIR / filename
        plt.savefig(output_path, dpi=180, bbox_inches='tight')
        print(f"Saved: {output_path.name}")
        plt.close()

def analyze_target(df, target_column):
    """Analyze the target variable distribution."""
    print("\n" + "=" * 60)
    print(f"TARGET VARIABLE ANALYSIS ({target_column})")
    print("=" * 60)
    
    target_counts = df[target_column].value_counts()
    target_pct = df[target_column].value_counts(normalize=True) * 100
    
    print(f"\nClass Distribution:")
    print(f"  0 (No purchase): {target_counts.get(0, 0)} ({target_pct.get(0, 0):.1f}%)")
    print(f"  1 (Purchase):    {target_counts.get(1, 0)} ({target_pct.get(1, 0):.1f}%)")
    
    # Check for imbalance
    imbalance_ratio = target_counts.min() / target_counts.max()
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 0.5:
        print("Warning: Dataset is imbalanced. Consider using SMOTE or class weights.")
    
    return target_counts

def analyze_features(df, target_columns):
    """Analyze individual features."""
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    
    # Remove target and ID columns
    numeric_cols = [
        c for c in numeric_cols
        if c not in target_columns + [
            'CustomerID',
            'latitude',
            'longitude'
        ]
    ]

    # City names are intentionally excluded because the new geography uses
    # country-relative city classes instead of hundreds of raw categories.
    categorical_cols = [
        c for c in categorical_cols
        if c not in ['AccountNumber', 'city']
    ]
    
    print(f"\nNumeric Features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    
    return numeric_cols, categorical_cols

def plot_distributions(df, numeric_cols, output_suffix):
    """Plot distributions of numeric features."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:6]):
        ax = axes[idx]
        df[col].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    output_path = DATA_DIR / f"feature_distributions_{output_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path.name}")
    plt.close()

def plot_correlation_matrix(df, numeric_cols, target_column, output_suffix):
    """Plot a readable target-focused correlation matrix."""
    corr_matrix = df[numeric_cols + [target_column]].corr()
    target_corr = corr_matrix[target_column].drop(target_column).abs()
    top_features = target_corr.sort_values(ascending=False).head(12).index.tolist()
    focused_cols = top_features + [target_column]
    focused_corr = corr_matrix.loc[focused_cols, focused_cols]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        focused_corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title(f'Top Feature Correlations - {target_column}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path = DATA_DIR / f"correlation_matrix_{output_suffix}.png"
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    print(f"Saved: {output_path.name}")
    plt.close()

    print("\nTop Correlations with Target:")
    for feat in target_corr.sort_values(ascending=False).head(12).index:
        print(f"  {feat}: {corr_matrix.loc[feat, target_column]:.3f}")

def plot_feature_vs_target(df, numeric_cols, target_column, output_suffix):
    """Plot features grouped by target."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:6]):
        ax = axes[idx]
        df.groupby(target_column)[col].hist(bins=30, alpha=0.6, ax=ax, legend=True)
        ax.set_title(f'{col} by {target_column}')
        ax.set_xlabel(col)
        ax.legend(['No purchase', 'Purchase'])
    
    plt.tight_layout()
    output_path = DATA_DIR / f"features_by_target_{output_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path.name}")
    plt.close()

def plot_categorical_analysis(df, categorical_cols, target_column):
    """Analyze categorical features."""
    if not categorical_cols:
        print("\nNo categorical features to analyze.")
        return
    
    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 60)
    
    for col in categorical_cols[:2]:  # Limit to first 2
        print(f"\n{col}:")
        print(df[col].value_counts())
        
        # Churn rate by category
        purchase_rate = df.groupby(col)[target_column].mean() * 100
        print(f"\nPurchase Rate by {col}:")
        print(purchase_rate.sort_values(ascending=False))

def generate_summary(df, target_column):
    """Generate summary insights."""
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    purchase_rate = df[target_column].mean() * 100
    print(f"\nOverall Purchase Rate: {purchase_rate:.1f}%")
    
    # Customers at risk
    at_risk = df[df['days_since_last_purchase'] > 180]
    print(f"\nCustomers with more than 180 days since purchase: {len(at_risk)}")
    
    # High-value customers
    high_value = df[df['total_spent'] > df['total_spent'].quantile(0.75)]
    high_value_purchase = high_value[target_column].mean() * 100
    print(f"High-Value Customer Purchase Rate: {high_value_purchase:.1f}%")
    
    # The engineered dataset contains one-hot category columns instead of the
    # original favorite_category label.
    if 'favorite_category' in df.columns:
        category_value = df.groupby(
            'favorite_category', dropna=False
        )['total_spent'].mean()
        print("\nAverage Spend by Category:")
        print(category_value.sort_values(ascending=False))

def main():
    print("Loading data...")
    df = load_data()
    
    basic_info(df)
    analyze_cohorts(df)
    analyze_geography(df)

    # Model-specific EDA uses engineered features, including distance to the
    # nearest major city, and only customers with purchase history at cutoff.
    model_df = load_engineered_data()
    model_df = model_df[model_df['customer_cohort'].isin([
        'existing_returned',
        'existing_not_returned'
    ])].copy()
    print(f"\nModel population: {len(model_df)} existing customers")

    raw_model_df = df[df['customer_cohort'].isin([
        'existing_returned',
        'existing_not_returned'
    ])].copy()
    distance_columns = ['CustomerID', 'distance_to_major_city_km']
    raw_model_df = raw_model_df.merge(
        model_df[distance_columns],
        on='CustomerID',
        how='left'
    )

    target_columns = ['will_buy_soon', 'will_buy_again_6m']
    numeric_cols, categorical_cols = analyze_features(model_df, target_columns)
    plot_geographic_distributions(model_df)

    for target_column in target_columns:
        target_counts = analyze_target(model_df, target_column)
        output_suffix = target_column.replace('will_buy_', '')

        print(f"\nGenerating visualizations for {target_column}...")
        plot_distributions(model_df, numeric_cols, output_suffix)
        plot_correlation_matrix(model_df, numeric_cols, target_column, output_suffix)
        plot_feature_vs_target(model_df, numeric_cols, target_column, output_suffix)
        plot_categorical_analysis(model_df, categorical_cols, target_column)
        analyze_geography_by_target(raw_model_df, target_column)
        generate_summary(model_df, target_column)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print("Next step: Run 03_preprocessing.py")

if __name__ == "__main__":
    main()
