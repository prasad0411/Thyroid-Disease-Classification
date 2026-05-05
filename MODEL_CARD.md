# Model Card: Thyroid Disease Classification System

## Model Overview

| Field | Details |
|-------|---------|
| **Model Type** | Ensemble (XGBoost + Random Forest, Soft Voting) |
| **Task** | Multi-class classification (3 classes) |
| **Classes** | Negative (euthyroid), Hypothyroid, Hyperthyroid |
| **Framework** | scikit-learn, XGBoost |
| **Published** | Springer Conference Proceedings, March 2024 |
| **Deployed** | [thyroid-disease-classification.streamlit.app](https://thyroid-disease-classification.streamlit.app) |

## Performance Metrics

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 97.6% |
| **Macro F1-Score** | 0.95+ |
| **Minority Class Recall** | 93% (up from 68% pre-SMOTE) |
| **Features Used** | 12 (reduced from 19 via RFE) |
| **Training Samples** | 7,200 patients |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.98 | 0.99 | 0.98 |
| Hypothyroid | 0.96 | 0.93 | 0.94 |
| Hyperthyroid | 0.94 | 0.93 | 0.93 |

## Training Data

- **Source**: UCI Machine Learning Repository — Thyroid Disease Dataset
- **Size**: 7,200 patient records
- **Features**: 12 clinical features selected via Recursive Feature Elimination (RFE)
- **Key Features**: TSH, T3, T4, FTI, T4U, age, sex, on_thyroxine, on_antithyroid, sick, pregnant, thyroid_surgery
- **Class Distribution**: Imbalanced (negative class dominant, ~3:1 ratio)
- **Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) applied to training set only

## Intended Use

- **Primary Use**: Clinical decision support for thyroid disease screening
- **Users**: Healthcare providers, medical researchers, clinical data scientists
- **Context**: Assistive tool for initial screening — not intended as sole diagnostic method
- **Deployment**: Web-based Streamlit dashboard with real-time prediction

## Limitations

- **Single-center data**: Trained on one dataset; may not generalize across populations with different demographics or lab assay methods
- **No longitudinal tracking**: Classifies based on single-visit lab values; does not account for trends over time
- **Feature constraints**: Requires specific lab values (TSH, T3, T4); cannot classify with partial data
- **Subclinical cases**: Borderline/subclinical thyroid conditions may be less accurately classified
- **No medication interaction modeling**: Does not account for drug interactions affecting thyroid function tests

## Ethical Considerations

- **Explainability**: Every prediction includes SHAP values showing individual feature contributions, enabling clinicians to verify model reasoning against clinical knowledge
- **Counterfactual Analysis**: Differential diagnosis shows what changes would alter the prediction, supporting clinical decision-making
- **Transparency**: Model architecture, training process, and limitations are fully documented
- **Bias Mitigation**: SMOTE addresses class imbalance to prevent systematic underdiagnosis of minority conditions
- **Human-in-the-loop**: System explicitly states it is not a substitute for clinical judgment

## Explainability Features

1. **SHAP Waterfall Charts**: Per-prediction feature contribution visualization
2. **Counterfactual Differential Analysis**: Shows minimum changes needed to flip diagnosis
3. **Patient vs Population Comparison**: Z-score comparison against training data distribution
4. **Clinical Reports**: Natural language interpretation of predictions with reference ranges
5. **RAG-Powered Q&A**: Evidence-based answers grounded in 25 indexed medical documents

## Model Versioning

Models are saved with timestamps (e.g., `best_model_20260428_230351.pkl`) and include:
- Trained model artifact (.pkl)
- Feature scaler (.pkl)
- Label encoder (.pkl)
- Metadata with performance metrics (.json)

## Citation

```bibtex
@inproceedings{kanade2024thyroid,
  title={Classification and Diagnosis of Thyroid Disease Using XGBoost and SHAP},
  author={Kanade, Prasad},
  booktitle={Springer Conference Proceedings},
  year={2024},
  doi={10.1007/978-981-97-6106-7_9}
}
```

## Author

**Prasad Kanade** — MS Computer Science, Northeastern University
- GitHub: [prasad0411](https://github.com/prasad0411)
- Paper: [Springer Link](https://link.springer.com/chapter/10.1007/978-981-97-6106-7_9)
