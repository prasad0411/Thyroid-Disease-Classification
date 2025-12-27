"""
Thyroid Disease Classification System
Production Machine Learning Pipeline

Author: Prasad Kanade
Institution: Northeastern University
Contact: kanade.pra@northeastern.edu

System Components:
- XGBoost Gradient Boosting Classifier
- SMOTE-based Class Imbalance Handling
- SHAP Model Explainability Framework
- Recursive Feature Elimination (RFE)
- Ensemble Learning (Soft Voting)

Execution: python thyroid_ml_professional.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def cleanup_previous_artifacts():
    """
    Remove previous model artifacts and logs to ensure clean execution environment.
    Maintains directory structure while clearing outdated files.
    """
    print("\n" + "="*80)
    print("INITIALIZING CLEAN EXECUTION ENVIRONMENT")
    print("="*80)
    
    models_dir = Path('models')
    if models_dir.exists():
        for file in models_dir.glob('*'):
            file.unlink()
        print("✓ Removed previous model artifacts")
    
    log_file = Path('outputs/training.log')
    if log_file.exists():
        log_file.unlink()
        print("✓ Cleared previous execution logs")
    
    print("✓ Visualization files will be updated")
    print("\n" + "="*80)
    print("ENVIRONMENT READY FOR EXECUTION")
    print("="*80 + "\n")


def generate_medical_dataset(n_samples=7200):
    """
    Generate realistic thyroid disease dataset with medical feature correlations.
    
    Dataset Characteristics:
    - Sample size: 7,200 patient records
    - Features: 19 clinical and demographic attributes
    - Target classes: 3 (Negative, Hypothyroid, Hyperthyroid)
    - Class distribution: Imbalanced (reflecting real-world prevalence)
    
    Realistic Properties:
    - TSH correlation with disease state: 80% (reduced from theoretical maximum)
    - Measurement uncertainty: ±8% (simulating laboratory variance)
    - Overlapping hormone ranges across classes
    - Edge cases: 3% of samples with atypical presentations
    
    Returns:
        pd.DataFrame: Complete dataset with features and target variable
    """
    np.random.seed(42)
    
    print(f"\n{'='*80}")
    print(f"DATASET GENERATION: {n_samples:,} PATIENT RECORDS")
    print(f"{'='*80}")
    print("\nIncorporating Clinical Realism:")
    print("  • Laboratory measurement uncertainty (±8%)")
    print("  • Overlapping hormone reference ranges")
    print("  • Subclinical and borderline presentations")
    print("  • Treatment status variability\n")
    
    # Generate primary diagnostic indicator (TSH)
    TSH_base = np.random.lognormal(0.5, 1.4, n_samples)
    
    # Assign diagnostic labels based on TSH levels with clinical uncertainty
    targets = []
    for tsh in TSH_base:
        if tsh > 5.0:  # Elevated TSH
            targets.append(np.random.choice(
                ['hypothyroid', 'negative', 'hyperthyroid'], 
                p=[0.80, 0.15, 0.05]
            ))
        elif tsh < 0.5:  # Suppressed TSH
            targets.append(np.random.choice(
                ['hyperthyroid', 'negative', 'hypothyroid'], 
                p=[0.75, 0.20, 0.05]
            ))
        else:  # Normal TSH range
            targets.append(np.random.choice(
                ['negative', 'hypothyroid', 'hyperthyroid'], 
                p=[0.92, 0.05, 0.03]
            ))
    
    targets = np.array(targets)
    
    # Apply measurement uncertainty to TSH
    TSH = TSH_base * np.random.normal(1.0, 0.08, n_samples)
    TSH = np.clip(TSH, 0.01, 50)
    
    # Generate T3 (Triiodothyronine) with clinically realistic distributions
    T3 = np.zeros(n_samples)
    for i, target in enumerate(targets):
        if target == 'hypothyroid':
            T3[i] = np.clip(np.random.normal(1.0, 0.4), 0.4, 1.8)
        elif target == 'hyperthyroid':
            T3[i] = np.clip(np.random.normal(4.2, 0.8), 2.5, 6.5)
        else:
            T3[i] = np.clip(np.random.normal(1.8, 0.6), 0.8, 3.5)
    
    T3 = T3 * np.random.normal(1.0, 0.08, n_samples)
    
    # Generate T4 (Thyroxine) with overlapping ranges
    T4 = np.zeros(n_samples)
    for i, target in enumerate(targets):
        if target == 'hypothyroid':
            T4[i] = np.clip(np.random.normal(70, 25), 35, 120)
        elif target == 'hyperthyroid':
            T4[i] = np.clip(np.random.normal(155, 30), 110, 230)
        else:
            T4[i] = np.clip(np.random.normal(105, 25), 60, 160)
    
    T4 = T4 * np.random.normal(1.0, 0.08, n_samples)
    
    # Additional clinical parameters
    T4U = np.clip(np.random.normal(1.0, 0.18, n_samples), 0.5, 1.8)
    
    # Treatment status (imperfect correlation with diagnosis)
    on_thyroxine = np.zeros(n_samples)
    on_antithyroid = np.zeros(n_samples)
    
    for i, target in enumerate(targets):
        if target == 'hypothyroid':
            on_thyroxine[i] = np.random.choice([0, 1], p=[0.40, 0.60])
        if target == 'hyperthyroid':
            on_antithyroid[i] = np.random.choice([0, 1], p=[0.45, 0.55])
    
    # Construct feature matrix
    data = {
        'age': np.clip(np.random.normal(48, 18, n_samples), 18, 90),
        'sex': np.random.choice([0, 1], n_samples, p=[0.68, 0.32]),
        'TSH': TSH,
        'T3': T3,
        'T4': T4,
        'T4U': T4U,
        'FTI': T4 / (T4U + 0.01),
        'on_thyroxine': on_thyroxine,
        'on_antithyroid': on_antithyroid,
        'sick': np.random.choice([0, 1], n_samples, p=[0.78, 0.22]),
        'pregnant': np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),
        'thyroid_surgery': np.random.choice([0, 1], n_samples, p=[0.89, 0.11]),
        'goitre': np.random.choice([0, 1], n_samples, p=[0.83, 0.17]),
        'tumor': np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),
        'query_hypothyroid': (targets == 'hypothyroid').astype(int) * np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'query_hyperthyroid': (targets == 'hyperthyroid').astype(int) * np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'TSH_measured': np.ones(n_samples),
        'T3_measured': np.random.choice([0, 1], n_samples, p=[0.12, 0.88]),
        'T4_measured': np.ones(n_samples),
        'target': targets
    }
    
    df = pd.DataFrame(data)
    
    # Introduce edge cases (atypical presentations)
    n_edge_cases = int(0.03 * n_samples)
    edge_indices = np.random.choice(n_samples, n_edge_cases, replace=False)
    
    for idx in edge_indices:
        current_target = df.loc[idx, 'target']
        if current_target == 'hypothyroid':
            df.loc[idx, 'target'] = np.random.choice(['negative', 'hyperthyroid'], p=[0.8, 0.2])
        elif current_target == 'hyperthyroid':
            df.loc[idx, 'target'] = np.random.choice(['negative', 'hypothyroid'], p=[0.7, 0.3])
    
    print(f"Dataset Generation Complete: {len(df):,} samples")
    print(f"\nClass Distribution:")
    for target, count in df['target'].value_counts().items():
        print(f"  {target:15} {count:5} ({count/len(df)*100:5.1f}%)")
    
    print(f"\nDataset Characteristics:")
    print(f"  TSH-Disease Correlation: 80%")
    print(f"  Measurement Uncertainty: ±8%")
    print(f"  Overlapping Ranges: Present")
    print(f"  Edge Cases: {n_edge_cases} samples ({n_edge_cases/n_samples*100:.1f}%)")
    print(f"\n{'='*80}\n")
    
    return df


# ==================== INITIALIZATION ====================
cleanup_previous_artifacts()

Path('models').mkdir(exist_ok=True)
Path('outputs/plots').mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

print("\n" + "="*80)
print("THYROID DISEASE CLASSIFICATION SYSTEM")
print("Production Machine Learning Pipeline")
print("="*80)
print("\nDeveloper: Prasad Kanade")
print("Institution: Northeastern University")
print("Techniques: XGBoost, SMOTE, SHAP, RFE")
print("="*80 + "\n")

# ==================== PIPELINE EXECUTION ====================

# Step 1: Data Acquisition
logger.info("="*80)
logger.info("PIPELINE STAGE 1: DATA ACQUISITION AND PREPARATION")
logger.info("="*80)

df = generate_medical_dataset(7200)
X = df.drop('target', axis=1)
y = df['target']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

logger.info(f"Dataset loaded successfully: {len(X):,} samples, {len(X.columns)} features")
logger.info(f"Target classes: {list(label_encoder.classes_)}")
logger.info(f"Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

# Step 2: Feature Selection
logger.info("\n" + "="*80)
logger.info("PIPELINE STAGE 2: FEATURE SELECTION (RFE)")
logger.info("="*80)

estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
selector = RFE(estimator, n_features_to_select=12, step=1)
selector.fit(X, y_encoded)

selected_features = X.columns[selector.support_].tolist()
X_selected = X[selected_features]

logger.info(f"Dimensionality reduction: {len(X.columns)} features → {len(selected_features)} features")
logger.info(f"Selected features: {selected_features}")

# Step 3: Data Partitioning
logger.info("\n" + "="*80)
logger.info("PIPELINE STAGE 3: TRAIN-TEST PARTITIONING")
logger.info("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

logger.info(f"Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
logger.info(f"Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
logger.info("Stratification: Applied to preserve class distribution")

# Step 4: Feature Normalization
logger.info("\n" + "="*80)
logger.info("PIPELINE STAGE 4: FEATURE SCALING")
logger.info("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logger.info("Normalization method: StandardScaler (zero mean, unit variance)")
logger.info("Training set: Fitted and transformed")
logger.info("Test set: Transformed using training statistics")

# Step 5: Class Imbalance Mitigation
if SMOTE_AVAILABLE:
    logger.info("\n" + "="*80)
    logger.info("PIPELINE STAGE 5: CLASS IMBALANCE HANDLING (SMOTE)")
    logger.info("="*80)
    
    logger.info(f"Pre-balancing distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    logger.info(f"Post-balancing distribution: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")
    logger.info(f"Synthetic sample generation: {len(X_train_scaled):,} → {len(X_train_balanced):,} samples")
else:
    X_train_balanced, y_train_balanced = X_train_scaled, y_train
    logger.warning("SMOTE unavailable - proceeding without class balancing")

# Step 6: Model Training
logger.info("\n" + "="*80)
logger.info("PIPELINE STAGE 6: MODEL TRAINING")
logger.info("="*80)

models = {}

if XGB_AVAILABLE:
    logger.info("Initializing XGBoost Classifier...")
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    models['XGBoost'].fit(X_train_balanced, y_train_balanced)
    logger.info("XGBoost training completed successfully")

logger.info("Initializing Random Forest Classifier...")
models['RandomForest'] = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
models['RandomForest'].fit(X_train_balanced, y_train_balanced)
logger.info("Random Forest training completed successfully")

if len(models) > 1:
    logger.info("Creating Ensemble Model (Soft Voting)...")
    models['Ensemble'] = VotingClassifier(
        estimators=list(models.items()),
        voting='soft',
        n_jobs=-1
    )
    models['Ensemble'].fit(X_train_balanced, y_train_balanced)
    logger.info("Ensemble model training completed successfully")

logger.info(f"Total models trained: {len(models)}")

# Step 7: Model Evaluation
logger.info("\n" + "="*80)
logger.info("PIPELINE STAGE 7: MODEL EVALUATION")
logger.info("="*80)

results = {}

for model_name, model in models.items():
    y_predictions = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_predictions, average='weighted', zero_division=0
    )
    
    logger.info(f"\n{model_name} Performance Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    logger.info(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    logger.info(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Per-class performance
    report = classification_report(
        y_test, y_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    logger.info(f"\nPer-Class Performance ({model_name}):\n{report}")
    
    # Generate confusion matrix visualization
    cm = confusion_matrix(y_test, y_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate feature importance visualization
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
        plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
        plt.xlabel('Feature Importance Score', fontsize=11)
        plt.title(f'Feature Importance Analysis: {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'outputs/plots/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Model comparison visualization
results_df = pd.DataFrame(results).T

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
plt.savefig('outputs/plots/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Step 8: Model Explainability
if SHAP_AVAILABLE and 'XGBoost' in models:
    logger.info("\n" + "="*80)
    logger.info("PIPELINE STAGE 8: MODEL EXPLAINABILITY (SHAP)")
    logger.info("="*80)
    
    try:
        explainer = shap.TreeExplainer(models['XGBoost'])
        shap_values = explainer.shap_values(X_test_scaled[:100])
        
        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            shap_values, X_test_scaled[:100],
            feature_names=selected_features,
            plot_type="bar",
            show=False
        )
        plt.title('SHAP Feature Importance Analysis: XGBoost', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/plots/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("SHAP explainability analysis completed")
        logger.info("Visualizations generated for clinical interpretation")
    except Exception as e:
        logger.warning(f"SHAP analysis encountered error: {str(e)}")

# Step 9: Model Persistence
logger.info("\n" + "="*80)
logger.info("PIPELINE STAGE 9: MODEL PERSISTENCE")
logger.info("="*80)

best_model_name = max(results, key=lambda k: results[k]['f1_score'])
best_model = models[best_model_name]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save trained model
joblib.dump(best_model, f'models/best_model_{timestamp}.pkl')
logger.info(f"Best performing model saved: {best_model_name}")

# Save preprocessing artifacts
joblib.dump(scaler, f'models/scaler_{timestamp}.pkl')
joblib.dump(label_encoder, f'models/label_encoder_{timestamp}.pkl')
logger.info("Feature scaler and label encoder persisted")

# Save execution metadata
metadata = {
    'execution_timestamp': timestamp,
    'best_model': best_model_name,
    'dataset_size': len(df),
    'features_original': 19,
    'features_selected': len(selected_features),
    'selected_features': selected_features,
    'target_classes': list(label_encoder.classes_),
    'performance_metrics': {
        model: {metric: float(value) for metric, value in metrics.items()}
        for model, metrics in results.items()
    }
}

import json
with open(f'models/metadata_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=2)

logger.info(f"Execution metadata saved: metadata_{timestamp}.json")
logger.info(f"Best model F1-score: {results[best_model_name]['f1_score']:.4f}")

# Final Summary
logger.info("\n" + "="*80)
logger.info("EXECUTION SUMMARY")
logger.info("="*80)

print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print(f"\n{results_df.to_string()}")
print(f"\nOptimal Model: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f} ({results[best_model_name]['f1_score']*100:.2f}%)")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f} ({results[best_model_name]['accuracy']*100:.2f}%)")
print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("="*80)
print("\nOutput Locations:")
print(f"  Visualizations: outputs/plots/ ({len(list(Path('outputs/plots').glob('*.png')))} files)")
print(f"  Models: models/ (timestamp: {timestamp})")
print(f"  Execution log: outputs/training.log")
print("\n" + "="*80 + "\n")

logger.info("="*80)
logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
logger.info("="*80)