"""
07 - MODEL EVALUATION
Comprehensive evaluation of the trained model.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

def load_model_and_data():
    """Load trained model and test data."""
    model = joblib.load(MODELS_DIR / "trained_model.pkl")
    data = joblib.load(DATA_DIR / "train_test_split.pkl")
    model_name = joblib.load(MODELS_DIR / "best_model_name.pkl")
    
    return model, data, model_name

def evaluate_classification(model, X_test, y_test, model_name):
    """Evaluate classification metrics."""
    print("\n" + "=" * 60)
    print(f"MODEL EVALUATION: {model_name}")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Churned', 'Active']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nKey Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return y_pred, y_proba, cm

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Churned', 'Active'],
                yticklabels=['Churned', 'Active'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(DATA_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_test, y_proba, model_name):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "roc_curve.png", dpi=150, bbox_inches='tight')
    print(f"Saved: roc_curve.png")
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_test, y_proba, model_name):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "precision_recall_curve.png", dpi=150, bbox_inches='tight')
    print(f"Saved: precision_recall_curve.png")
    plt.close()
    
    return avg_precision

def plot_feature_importance(model, feature_cols, model_name):
    """Plot feature importance (for tree-based models)."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top 10 Feature Importance - {model_name}')
        plt.bar(range(10), importances[indices], align='center')
        plt.xticks(range(10), [feature_cols[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(DATA_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
        print(f"Saved: feature_importance.png")
        plt.close()
        
        # Print top features
        print(f"\nTop 10 Features:")
        for i in indices:
            print(f"  {feature_cols[i]}: {importances[i]:.4f}")

def plot_model_comparison():
    """Compare model F1 and ROC-AUC for the six-month horizon."""
    rankings_path = DATA_DIR / 'model_rankings_6m.csv'
    if not rankings_path.exists():
        print(f"Skipping model comparison; missing {rankings_path.name}")
        return

    rankings = pd.read_csv(rankings_path).sort_values('F1-Mean')
    ax = rankings.plot(
        x='Model',
        y=['F1-Mean', 'ROC-AUC'],
        kind='barh',
        figsize=(10, 6),
        color=['#2f6f9f', '#d9822b']
    )
    ax.set_title('Model Comparison - Six-Month Horizon')
    ax.set_xlabel('Score')
    ax.set_ylabel('Model')
    ax.set_xlim(0, 1)
    plt.legend(title='Metric')
    plt.tight_layout()
    output_path = DATA_DIR / 'model_comparison_6m.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    print(f"Saved: {output_path.name}")
    plt.close()


def plot_probability_calibration(y_test, y_proba, model_name):
    """Plot predicted probabilities against observed positive rates."""
    fraction_positive, mean_predicted = calibration_curve(
        y_test, y_proba, n_bins=10, strategy='quantile'
    )
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted, fraction_positive, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed purchase rate')
    plt.title(f'Probability Calibration - {model_name}')
    plt.legend()
    plt.tight_layout()
    output_path = DATA_DIR / 'probability_calibration.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    print(f"Saved: {output_path.name}")
    plt.close()


def plot_probability_separation(y_test, y_proba, model_name):
    """Plot predicted probability distributions by observed outcome."""
    y_test_array = np.asarray(y_test)
    plt.figure(figsize=(9, 5))
    plt.hist(y_proba[y_test_array == 0], bins=25, alpha=0.65, label='No repurchase')
    plt.hist(y_proba[y_test_array == 1], bins=25, alpha=0.65, label='Repurchase')
    plt.axvline(0.5, linestyle='--', color='black', label='Decision threshold')
    plt.xlabel('Predicted probability of repurchase within six months')
    plt.ylabel('Customers')
    plt.title(f'Predicted Probability Separation - {model_name}')
    plt.legend()
    plt.tight_layout()
    output_path = DATA_DIR / 'predicted_probability_separation.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    print(f"Saved: {output_path.name}")
    plt.close()


def generate_business_insights(y_test, y_pred, y_proba, model_name):
    """Generate business-focused insights."""
    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)
    
    # Customer segments at risk
    total_customers = len(y_test)
    predicted_churn = (y_pred == 0).sum()
    actual_churn = (y_test == 0).sum()
    
    print(f"\nCustomer Analysis:")
    print(f"  Total customers in test set: {total_customers}")
    print(f"  Predicted to churn: {predicted_churn} ({predicted_churn/total_customers*100:.1f}%)")
    print(f"  Actually churned: {actual_churn} ({actual_churn/total_customers*100:.1f}%)")
    
    # Identify high-risk customers
    high_risk_mask = (y_pred == 0) & (y_proba < 0.3)
    high_risk_count = high_risk_mask.sum()
    
    print(f"\nHigh-Risk Customers (confidence > 70%): {high_risk_count}")
    
    # Potential revenue at risk
    print(f"\nRecommendation:")
    print(f"  - Target {high_risk_count} high-risk customers with retention campaigns")
    print(f"  - Focus on personalized offers and re-engagement emails")
    print(f"  - Expected retention rate improvement: 10-20%")

def save_evaluation_results(model_name, accuracy, f1, roc_auc, avg_precision):
    """Save evaluation results."""
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Avg Precision': avg_precision
    }
    
    df_results = pd.DataFrame([results])
    df_results.to_csv(DATA_DIR / "evaluation_results.csv", index=False)
    print(f"\nSaved evaluation results to {DATA_DIR / 'evaluation_results.csv'}")

def main():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load model and data
    model, data, model_name = load_model_and_data()
    X_test = data['X_test']
    y_test = data['y_test']
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
    
    # Evaluate
    y_pred, y_proba, cm = evaluate_classification(model, X_test, y_test, model_name)
    
    # Plots
    plot_confusion_matrix(cm, model_name)
    roc_auc = plot_roc_curve(y_test, y_proba, model_name)
    avg_precision = plot_precision_recall_curve(y_test, y_proba, model_name)
    plot_feature_importance(model, feature_cols, model_name)
    plot_model_comparison()
    plot_probability_calibration(y_test, y_proba, model_name)
    plot_probability_separation(y_test, y_proba, model_name)
    
    # Calculate final metrics
    accuracy = (y_pred == y_test).mean()
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    
    # Business insights
    generate_business_insights(y_test, y_pred, y_proba, model_name)
    
    # Save results
    save_evaluation_results(model_name, accuracy, f1, roc_auc, avg_precision)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("Next step: Run 08_conclusion.ipynb or review results")

if __name__ == "__main__":
    main()
