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

def analyze_target(df):
    """Analyze the target variable distribution."""
    print("\n" + "=" * 60)
    print("TARGET VARIABLE ANALYSIS (will_buy_again)")
    print("=" * 60)
    
    target_counts = df['will_buy_again'].value_counts()
    target_pct = df['will_buy_again'].value_counts(normalize=True) * 100
    
    print(f"\nClass Distribution:")
    print(f"  0 (Churned): {target_counts[0]} ({target_pct[0]:.1f}%)")
    print(f"  1 (Active):  {target_counts[1]} ({target_pct[1]:.1f}%)")
    
    # Check for imbalance
    imbalance_ratio = target_counts.min() / target_counts.max()
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 0.5:
        print("Warning: Dataset is imbalanced. Consider using SMOTE or class weights.")
    
    return target_counts

def analyze_features(df):
    """Analyze individual features."""
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target and ID columns
    numeric_cols = [c for c in numeric_cols if c not in ['will_buy_again', 'CustomerID']]

    categorical_cols = [c for c in categorical_cols if c != 'AccountNumber']
    
    print(f"\nNumeric Features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    
    return numeric_cols, categorical_cols

def plot_distributions(df, numeric_cols):
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
    plt.savefig(DATA_DIR / "feature_distributions.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: feature_distributions.png")
    plt.close()

def plot_correlation_matrix(df, numeric_cols):
    """Plot correlation matrix."""
    # Add target to correlation
    cols = numeric_cols + ['will_buy_again']
    
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(DATA_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
    print(f"Saved: correlation_matrix.png")
    plt.close()
    
    # Print top correlations with target
    target_corr = corr_matrix['will_buy_again'].drop('will_buy_again').abs().sort_values(ascending=False)
    print("\nTop Correlations with Target:")
    for feat, corr in target_corr.items():
        print(f"  {feat}: {corr:.3f}")

def plot_feature_vs_target(df, numeric_cols):
    """Plot features grouped by target."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:6]):
        ax = axes[idx]
        df.groupby('will_buy_again')[col].hist(bins=30, alpha=0.6, ax=ax, legend=True)
        ax.set_title(f'{col} by Churn Status')
        ax.set_xlabel(col)
        ax.legend(['Churned', 'Active'])
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / "features_by_target.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: features_by_target.png")
    plt.close()

def plot_categorical_analysis(df, categorical_cols):
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
        churn_rate = df.groupby(col)['will_buy_again'].mean() * 100
        print(f"\nChurn Rate by {col}:")
        print(churn_rate.sort_values(ascending=False))

def generate_summary(df):
    """Generate summary insights."""
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # Overall churn rate
    churn_rate = (1 - df['will_buy_again'].mean()) * 100
    print(f"\nOverall Churn Rate: {churn_rate:.1f}%")
    
    # Customers at risk
    at_risk = df[df['days_since_last_purchase'] > 180]
    print(f"\nCustomers at Risk (no purchase in 6 months): {len(at_risk)}")
    
    # High-value customers
    high_value = df[df['total_spent'] > df['total_spent'].quantile(0.75)]
    high_value_churn = (1 - high_value['will_buy_again'].mean()) * 100
    print(f"High-Value Customer Churn Rate: {high_value_churn:.1f}%")
    
    # Most valuable category
    category_value = df.groupby('favorite_category')['total_spent'].mean()
    print(f"\nAverage Spend by Category:")
    print(category_value.sort_values(ascending=False))

def main():
    print("Loading data...")
    df = load_data()
    
    basic_info(df)
    target_counts = analyze_target(df)
    numeric_cols, categorical_cols = analyze_features(df)
    
    print("\nGenerating visualizations...")
    plot_distributions(df, numeric_cols)
    plot_correlation_matrix(df, numeric_cols)
    plot_feature_vs_target(df, numeric_cols)
    plot_categorical_analysis(df, categorical_cols)
    
    generate_summary(df)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print("Next step: Run 03_preprocessing.py")

if __name__ == "__main__":
    main()
