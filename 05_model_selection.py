"""
05 - MODEL SELECTION
Compare multiple algorithms to find the best one.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

# warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

def load_data(target_column):
    """Load engineered features."""
    df = pd.read_csv(DATA_DIR / "engineered_features.csv")

    # Model only repeat-purchase behavior from customers who existed at the cutoff.
    df = df[df['customer_cohort'].isin([
        'existing_returned',
        'existing_not_returned'
    ])].copy()
    
    # Separate features and target
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
    
    X = df[feature_cols]
    y = df[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    print(f"Target: {target_column}")
    return X, y

def get_models():
    """Define models to compare."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    return models

def evaluate_models(X, y, models):
    """Evaluate models using cross-validation."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (5-Fold Cross-Validation)")
    print("=" * 60)
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cada modelo se evalúa mediante validación cruzada en 5 folds.
    # Para cada modelo se calcula y almacena cada métrica individual;
    # después, todas sus métricas se agrupan dentro de results.
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Multiple metrics
        metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            metrics[metric] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
        
        results[name] = metrics
        
        # Print results
        print(f"  Accuracy:  {metrics['accuracy']['mean']:.4f} (+/- {metrics['accuracy']['std']:.4f})")
        print(f"  Precision: {metrics['precision']['mean']:.4f} (+/- {metrics['precision']['std']:.4f})")
        print(f"  Recall:    {metrics['recall']['mean']:.4f} (+/- {metrics['recall']['std']:.4f})")
        print(f"  F1-Score:  {metrics['f1']['mean']:.4f} (+/- {metrics['f1']['std']:.4f})")
        print(f"  ROC-AUC:   {metrics['roc_auc']['mean']:.4f} (+/- {metrics['roc_auc']['std']:.4f})")
    
    return results

def rank_models(results):
    """Rank models by F1-score."""
    print("\n" + "=" * 60)
    print("MODEL RANKING (by F1-Score)")
    print("=" * 60)
    
    rankings = []
    for name, metrics in results.items():
        rankings.append({
            'Model': name,
            'F1-Mean': metrics['f1']['mean'],
            'F1-Std': metrics['f1']['std'],
            'Accuracy': metrics['accuracy']['mean'],
            'ROC-AUC': metrics['roc_auc']['mean']
        })
    
    df_rankings = pd.DataFrame(rankings)
    df_rankings = df_rankings.sort_values('F1-Mean', ascending=False)
    
    print("\n" + df_rankings.to_string(index=False))
    
    # Best model
    best_model_name = df_rankings.iloc[0]['Model']
    best_f1 = df_rankings.iloc[0]['F1-Mean']
    
    print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")
    
    return df_rankings, best_model_name

def select_best_model(best_model_name, models):
    """Return the best model instance."""
    return models[best_model_name]

def main():
    print("=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    
    targets = {
        'soon': 'will_buy_soon',
        '6m': 'will_buy_again_6m'
    }

    for horizon, target_column in targets.items():
        print(f"\nEvaluating horizon: {horizon}")
        X, y = load_data(target_column)
        models = get_models()
        results = evaluate_models(X, y, models)
        rankings, best_model_name = rank_models(results)

        rankings_path = DATA_DIR / f"model_rankings_{horizon}.csv"
        rankings.to_csv(rankings_path, index=False)
        joblib.dump(best_model_name, MODELS_DIR / f"best_model_name_{horizon}.pkl")
        print(f"Saved rankings to {rankings_path}")
        print(f"Saved best model name: {best_model_name}")

        if horizon == '6m':
            rankings.to_csv(DATA_DIR / "model_rankings.csv", index=False)
            joblib.dump(best_model_name, MODELS_DIR / "best_model_name.pkl")
    
    print("\n" + "=" * 60)
    print("MODEL SELECTION COMPLETE")
    print("=" * 60)
    print("Next step: Run 06_training.py")

if __name__ == "__main__":
    main()
