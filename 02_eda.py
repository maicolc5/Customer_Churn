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
    numeric_cols = [c for c in numeric_cols if c not in target_columns + ['CustomerID']]

    categorical_cols = [c for c in categorical_cols if c != 'AccountNumber']
    
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
    """Plot correlation matrix."""
    # Add target to correlation
    cols = numeric_cols + [target_column]
    
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    output_path = DATA_DIR / f"correlation_matrix_{output_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path.name}")
    plt.close()
    
    # Print top correlations with target
    target_corr = corr_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)
    print("\nTop Correlations with Target:")
    for feat, corr in target_corr.items():
        print(f"  {feat}: {corr:.3f}")

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
    
    # Most valuable category
    category_value = df.groupby('favorite_category', dropna=False)['total_spent'].mean()
    print(f"\nAverage Spend by Category:")
    print(category_value.sort_values(ascending=False))

def main():
    print("Loading data...")
    df = load_data()
    
    basic_info(df)
    analyze_cohorts(df)

    # Model-specific EDA: only customers with purchase history before the cutoff.
    model_df = df[df['customer_cohort'].isin([
        'existing_returned',
        'existing_not_returned'
    ])].copy()
    print(f"\nModel population: {len(model_df)} existing customers")

    target_columns = ['will_buy_soon', 'will_buy_again_6m']
    numeric_cols, categorical_cols = analyze_features(model_df, target_columns)

    for target_column in target_columns:
        target_counts = analyze_target(model_df, target_column)
        output_suffix = target_column.replace('will_buy_', '')

        print(f"\nGenerating visualizations for {target_column}...")
        plot_distributions(model_df, numeric_cols, output_suffix)
        plot_correlation_matrix(model_df, numeric_cols, target_column, output_suffix)
        plot_feature_vs_target(model_df, numeric_cols, target_column, output_suffix)
        plot_categorical_analysis(model_df, categorical_cols, target_column)
        generate_summary(model_df, target_column)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print("Next step: Run 03_preprocessing.py")

if __name__ == "__main__":
    main()
