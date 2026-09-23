"""
06 - MODEL TRAINING
Train the best model with optimized hyperparameters.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

def load_data():
    """Load engineered features."""
    df = pd.read_csv(DATA_DIR / "engineered_features.csv")
    df = df[df['customer_cohort'].isin([
        'existing_returned',
        'existing_not_returned'
    ])].copy()
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
    
    X = df[feature_cols]
    y = df['will_buy_again_6m']
    
    return X, y, feature_cols

def split_data(X, y):
    """Split data into train and test sets."""
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    print(f"\nTrain target distribution:\n{y_train.value_counts()}")
    print(f"\nTest target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test

def get_model_and_params(model_name):
    """Get model and hyperparameter grid based on best model name."""
    models_and_params = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [5, 10, 20]
            }
        },
        'Logistic Regression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(random_state=42, max_iter=2000))
            ]),
            'params': {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga']
            }
        }
    }
    if model_name not in models_and_params:
        available_models = ', '.join(models_and_params)
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available_models}"
        )

    return models_and_params[model_name]

def train_with_gridsearch(X_train, y_train, model_name):
    """Train model with grid search cross-validation."""
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    
    model_info = get_model_and_params(model_name)
    model = model_info['model']
    param_grid = model_info['params']
    
    print(f"\nHyperparameter Grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    print("\nRunning Grid Search (this may take a while)...")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV F1-Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_final_model(X_train, y_train, model, model_name):
    """Train the final model on full training data."""
    print("\nTraining final model on full training data...")
    model.fit(X_train, y_train)
    print("Training complete!")
    return model

def save_model(model, feature_cols, model_name):
    """Save trained model and metadata."""
    # Save model
    joblib.dump(model, MODELS_DIR / "trained_model.pkl")
    print(f"\nSaved trained model to {MODELS_DIR / 'trained_model.pkl'}")
    
    # Save feature columns
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")
    print(f"Saved feature columns to {MODELS_DIR / 'feature_columns.pkl'}")
    
    # Save model name
    joblib.dump(model_name, MODELS_DIR / "best_model_name.pkl")
    print(f"Saved model name: {model_name}")

def main():
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    X, y, feature_cols = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Get best model name
    best_model_name = joblib.load(MODELS_DIR / "best_model_name.pkl")
    print(f"\nTraining best model: {best_model_name}")
    
    # Train with grid search
    best_model, best_params = train_with_gridsearch(X_train, y_train, best_model_name)
    
    # Train final model
    final_model = train_final_model(X_train, y_train, best_model, best_model_name)
    
    # Save
    save_model(final_model, feature_cols, best_model_name)
    
    # Save test data for evaluation
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, DATA_DIR / "train_test_split.pkl")
    print("Saved train/test split for evaluation")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Next step: Run 07_evaluation.py")

if __name__ == "__main__":
    main()
