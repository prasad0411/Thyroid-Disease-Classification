"""
Thyroid Disease Classification System - Main Training Pipeline
Production Machine Learning Pipeline

Author: Prasad Kanade
Institution: Northeastern University
Contact: kanade.pra@northeastern.edu

Execution: python train.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Import project modules
from config import *
from data_generator import generate_medical_dataset
from utils import (cleanup_previous_artifacts, setup_directories,
                  plot_confusion_matrix, plot_feature_importance,
                  plot_model_comparison, plot_shap_summary)

# Optional dependencies
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(OUTPUTS_DIR / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def main():
    """Main training pipeline"""
    
    # Initialize
    cleanup_previous_artifacts()
    setup_directories()
    logger = setup_logging()
    
    print("\n" + "="*80)
    print("THYROID DISEASE CLASSIFICATION SYSTEM")
    print("Production Machine Learning Pipeline")
    print("="*80)
    print("\nDeveloper: Prasad Kanade")
    print("Institution: Northeastern University")
    print("Techniques: XGBoost, SMOTE, SHAP, RFE")
    print("="*80 + "\n")
    
    # Step 1: Data Acquisition
    logger.info("="*80)
    logger.info("PIPELINE STAGE 1: DATA ACQUISITION AND PREPARATION")
    logger.info("="*80)
    
    df = generate_medical_dataset()
    X = df.drop('target', axis=1)
    y = df['target']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Dataset loaded: {len(X):,} samples, {len(X.columns)} features")
    logger.info(f"Target classes: {list(label_encoder.classes_)}")
    
    # Step 2: Feature Selection
    logger.info("\n" + "="*80)
    logger.info("PIPELINE STAGE 2: FEATURE SELECTION (RFE)")
    logger.info("="*80)
    
    estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=N_FEATURES_SELECTED, step=1)
    selector.fit(X, y_encoded)
    
    selected_features = X.columns[selector.support_].tolist()
    X_selected = X[selected_features]
    
    logger.info(f"Dimensionality reduction: {len(X.columns)} → {len(selected_features)} features")
    logger.info(f"Selected features: {selected_features}")
    
    # Step 3: Train-Test Split
    logger.info("\n" + "="*80)
    logger.info("PIPELINE STAGE 3: TRAIN-TEST PARTITIONING")
    logger.info("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    
    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    
    # Step 4: Feature Scaling
    logger.info("\n" + "="*80)
    logger.info("PIPELINE STAGE 4: FEATURE SCALING")
    logger.info("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Normalization: StandardScaler applied")
    
    # Step 5: SMOTE
    if SMOTE_AVAILABLE:
        logger.info("\n" + "="*80)
        logger.info("PIPELINE STAGE 5: CLASS IMBALANCE HANDLING (SMOTE)")
        logger.info("="*80)
        
        logger.info(f"Pre-balancing: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto')
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        logger.info(f"Post-balancing: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        logger.warning("SMOTE unavailable - proceeding without balancing")
    
    # Step 6: Model Training
    logger.info("\n" + "="*80)
    logger.info("PIPELINE STAGE 6: MODEL TRAINING")
    logger.info("="*80)
    
    models = {}
    
    if XGB_AVAILABLE:
        logger.info("Training XGBoost...")
        models['XGBoost'] = xgb.XGBClassifier(**XGBOOST_PARAMS)
        models['XGBoost'].fit(X_train_balanced, y_train_balanced)
    
    logger.info("Training Random Forest...")
    models['RandomForest'] = RandomForestClassifier(**RANDOMFOREST_PARAMS)
    models['RandomForest'].fit(X_train_balanced, y_train_balanced)
    
    if len(models) > 1:
        logger.info("Creating Ensemble...")
        models['Ensemble'] = VotingClassifier(
            estimators=list(models.items()),
            voting='soft',
            n_jobs=-1
        )
        models['Ensemble'].fit(X_train_balanced, y_train_balanced)
    
    logger.info(f"Total models trained: {len(models)}")
    
    # Step 7: Model Evaluation
    logger.info("\n" + "="*80)
    logger.info("PIPELINE STAGE 7: MODEL EVALUATION")
    logger.info("="*80)
    
    results = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Visualizations
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, label_encoder.classes_, model_name)
        
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model.feature_importances_, selected_features, model_name)
    
    results_df = pd.DataFrame(results).T
    plot_model_comparison(results_df)
    
    # Step 8: SHAP Explainability
    if SHAP_AVAILABLE and 'XGBoost' in models:
        logger.info("\n" + "="*80)
        logger.info("PIPELINE STAGE 8: MODEL EXPLAINABILITY (SHAP)")
        logger.info("="*80)
        
        try:
            explainer = shap.TreeExplainer(models['XGBoost'])
            shap_values = explainer.shap_values(X_test_scaled[:100])
            plot_shap_summary(shap_values, X_test_scaled[:100], selected_features)
            logger.info("SHAP analysis completed")
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {str(e)}")
    
    # Step 9: Model Persistence
    logger.info("\n" + "="*80)
    logger.info("PIPELINE STAGE 9: MODEL PERSISTENCE")
    logger.info("="*80)
    
    best_model_name = max(results, key=lambda k: results[k]['f1_score'])
    best_model = models[best_model_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    joblib.dump(best_model, MODELS_DIR / f'best_model_{timestamp}.pkl')
    joblib.dump(scaler, MODELS_DIR / f'scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, MODELS_DIR / f'label_encoder_{timestamp}.pkl')
    
    metadata = {
        'execution_timestamp': timestamp,
        'best_model': best_model_name,
        'dataset_size': len(df),
        'features_selected': selected_features,
        'target_classes': list(label_encoder.classes_),
        'performance_metrics': {
            model: {metric: float(value) for metric, value in metrics.items()}
            for model, metrics in results.items()
        }
    }
    
    with open(MODELS_DIR / f'metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Final Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\n{results_df.to_string()}")
    print(f"\nOptimal Model: {best_model_name}")
    print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    logger.info("="*80)
    logger.info("PIPELINE EXECUTION COMPLETED")
    logger.info("="*80)


if __name__ == "__main__":
    main()