"""
Utility functions for Thyroid Disease Classification System
Includes visualization, cleanup, and helper functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import PLOTS_DIR, MODELS_DIR, OUTPUTS_DIR, PLOT_STYLE, PLOT_PALETTE, DPI

plt.style.use(PLOT_STYLE)
sns.set_palette(PLOT_PALETTE)


def cleanup_previous_artifacts():
    """Remove previous model artifacts and logs"""
    print("\n" + "="*80)
    print("INITIALIZING CLEAN EXECUTION ENVIRONMENT")
    print("="*80)
    
    if MODELS_DIR.exists():
        for file in MODELS_DIR.glob('*'):
            file.unlink()
        print("✓ Removed previous model artifacts")
    
    log_file = OUTPUTS_DIR / 'training.log'
    if log_file.exists():
        log_file.unlink()
        print("✓ Cleared previous execution logs")
    
    print("✓ Visualization files will be updated")
    print("\n" + "="*80)
    print("ENVIRONMENT READY FOR EXECUTION")
    print("="*80 + "\n")


def setup_directories():
    """Create necessary directories"""
    MODELS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, classes, model_name):
    """Generate confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classes,
               yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'confusion_matrix_{model_name}.png', dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances, feature_names, model_name):
    """Generate feature importance visualization"""
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance Score', fontsize=11)
    plt.title(f'Feature Importance Analysis: {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'feature_importance_{model_name}.png', dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_df):
    """Generate model comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        results_df[metric].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', width=0.6)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim([0.90, 1.02])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)
    
    plt.suptitle('Comparative Model Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'model_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_shap_summary(shap_values, X_test, feature_names):
    """Generate SHAP summary visualization"""
    try:
        import shap
        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.title('SHAP Feature Importance Analysis: XGBoost', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'shap_summary.png', dpi=DPI, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP visualization failed: {str(e)}")