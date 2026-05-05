# 🧬 Thyroid Disease Classification — Clinical Decision Support System

[![Streamlit](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit)](https://thyroid-disease-classification.streamlit.app/)
[![Paper](https://img.shields.io/badge/Published-Springer-blue)](https://link.springer.com/chapter/10.1007/978-981-97-6106-7_9)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![Model Card](https://img.shields.io/badge/Model_Card-Responsible_AI-green)](/MODEL_CARD.md)

A production-grade ML system for thyroid disease classification with explainable AI, RAG-powered clinical Q&A, and an interactive Streamlit dashboard. Published in Springer Conference Proceedings (97.6% accuracy, 7,200 patients).

---

## 🏗️ Architecture

```
Patient Data
  → Feature Selection (RFE: 19 → 12 features)
  → SMOTE Class Balancing (3:1 → 1:1)
  → XGBoost + Random Forest Ensemble (Soft Voting)
  → SHAP Explainability (per-patient feature contributions)
  → RAG Retrieval (ChromaDB + 25 medical documents)
  → LLM Report Generation (Claude / GPT-4 / Template fallback)
  → Streamlit Clinical Dashboard (5 pages, deployed)
```

## ✨ Features

### 🔬 Predict & Explain
- Real-time classification with color-coded diagnosis banners
- SVG confidence gauge with class probability breakdown
- Inline lab value indicators (elevated/low/normal vs reference ranges)
- SHAP waterfall charts showing per-feature contributions
- Counterfactual differential analysis: "what would need to change for a different diagnosis?"
- Patient vs population z-score comparison

### 📚 Clinical Q&A (RAG)
- Semantic search over 25 indexed PubMed-style medical documents
- ChromaDB vector store with sentence-transformer embeddings
- LLM-generated answers with source citations (template fallback when no API key)
- Suggested follow-up questions based on topic

### 📊 Model Performance
- Interactive model comparison across all trained classifiers
- Training visualization tabs (confusion matrix, feature importance, SHAP summary)
- Dataset class distribution and feature statistics

### 📁 Batch Prediction
- Upload CSV for multi-patient screening (100+ simultaneous)
- Input validation: type coercion, missing column detection, empty file handling
- Downloadable results with predictions and confidence scores

### ℹ️ About & Methodology
- Full system architecture explanation in plain English
- Link to Springer publication
- Responsible AI documentation (see [Model Card](MODEL_CARD.md))

## 📈 Experiment History

| Experiment | Model | Features | SMOTE | Accuracy | Minority Recall |
|-----------|-------|----------|-------|----------|----------------|
| Baseline | XGBoost | 19 | No | 94.3% | 68% |
| +SMOTE | XGBoost | 19 | Yes | 96.1% | 89% |
| +RFE | XGBoost | 12 | Yes | 96.8% | 91% |
| RandomForest | RF | 12 | Yes | 96.4% | 90% |
| **Ensemble (Final)** | **XGB+RF** | **12** | **Yes** | **97.6%** | **93%** |

Full experiment log: [`experiments.json`](experiments.json)

## 🚀 Deployment

**Live**: [thyroid-disease-classification.streamlit.app](https://thyroid-disease-classification.streamlit.app/)

### Run Locally

```bash
git clone https://github.com/prasad0411/Thyroid-Disease-Classification.git
cd Thyroid-Disease-Classification
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push to GitHub (models must be tracked in git)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → branch `main` → main file `app.py`
4. Deploy

### Optional: LLM-Powered Reports

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in Streamlit Cloud secrets for LLM-generated clinical reports. Without an API key, the system uses a clinically-validated template engine.

## 📂 Project Structure

```
├── app.py                    # Streamlit dashboard (891 lines, 5 pages)
├── train.py                  # Model training pipeline
├── data_generator.py         # Synthetic data generation
├── models/                   # Trained model artifacts (timestamped)
│   ├── best_model_*.pkl
│   ├── scaler_*.pkl
│   ├── label_encoder_*.pkl
│   └── metadata_*.json
├── rag/
│   ├── documents.py          # 25 medical literature abstracts
│   ├── indexer.py            # ChromaDB vector store builder
│   └── retriever.py          # Semantic search retrieval
├── llm/
│   ├── report_generator.py   # LLM clinical report generation
│   └── clinical_qa.py        # RAG-powered Q&A
├── api/
│   └── predict.py            # FastAPI prediction endpoint
├── MODEL_CARD.md             # Responsible AI documentation
├── experiments.json          # Experiment tracking log
├── .streamlit/config.toml    # Streamlit theme configuration
├── requirements.txt
└── packages.txt              # System dependencies for Streamlit Cloud
```

## 📄 Publication

**Classification and Diagnosis of Thyroid Disease Using XGBoost and SHAP**
*Springer Conference Proceedings, March 2024*
[Read Paper →](https://link.springer.com/chapter/10.1007/978-981-97-6106-7_9)

## 👨‍💻 Author

**Prasad Kanade** — MS Computer Science, Northeastern University
- [GitHub](https://github.com/prasad0411) · [LinkedIn](https://linkedin.com/in/prasad-kanade-/) · kanade.pra@northeastern.edu
- [Portfolio](https://prasad0411.github.io/Prasad-Portfolio/)

---

*This system is intended for research and educational purposes. It is not a substitute for professional medical judgment.*
