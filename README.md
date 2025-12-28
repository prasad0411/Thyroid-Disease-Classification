# Thyroid Disease Classification System

A machine learning pipeline for classifying thyroid conditions using ensemble methods, SMOTE balancing, and SHAP explainability.

## Author
**Prasad Kanade**  
Northeastern University  
kanade.pra@northeastern.edu

## Features
- **97.6% Classification Accuracy** across 3 thyroid conditions
- **XGBoost Ensemble Methods** with soft voting
- **SMOTE Class Balancing** for imbalanced medical data
- **SHAP Explainability** for clinical interpretability
- **Automated Feature Selection** (19 → 12 features via RFE)

## Project Structure
```
Thyroid-Disease-Classification/
├── config.py              # Configuration and parameters
├── data_generator.py      # Synthetic medical dataset generation
├── utils.py               # Visualization and helper functions
├── train.py               # Main training pipeline
├── requirements.txt       # Python dependencies
├── models/                # Saved models (generated)
└── outputs/               # Logs and visualizations (generated)
    └── plots/
```

## Installation
```bash
# Clone repository
git clone https://github.com/prasad0411/Thyroid-Disease-Classification.git
cd Thyroid-Disease-Classification

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run training pipeline
python train.py
```

## System Components

### Models
- **XGBoost Gradient Boosting Classifier**
- **Random Forest Classifier**
- **Ensemble Voting Classifier**

### Techniques
- **SMOTE:** Synthetic Minority Over-sampling
- **RFE:** Recursive Feature Elimination
- **SHAP:** Model explainability framework
- **StandardScaler:** Feature normalization

### Dataset
- **7,200 patient records** (synthetic, clinically realistic)
- **19 clinical features** (TSH, T3, T4, demographics, treatment history)
- **3 target classes:** Negative, Hypothyroid, Hyperthyroid

## Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 97.6% | 97.7% | 97.6% | 97.6% |
| Random Forest | 96.8% | 96.9% | 96.8% | 96.8% |
| Ensemble | 97.4% | 97.5% | 97.4% | 97.4% |

## Key Findings
- **TSH, T3, T4** hormones contribute **65% combined feature importance**
- **SMOTE balancing** improved minority class recall by **25%**
- **Feature reduction** (37%) maintained clinical interpretability

## Outputs
- **Models:** `models/best_model_*.pkl`
- **Visualizations:** `outputs/plots/*.png`
  - Confusion matrices
  - Feature importance plots
  - SHAP summary plots
  - Model comparison charts
- **Logs:** `outputs/training.log`
- **Metadata:** `models/metadata_*.json`
