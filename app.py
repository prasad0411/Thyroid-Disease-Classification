"""
Thyroid Disease Classification - Interactive Clinical Dashboard

Features:
1. Patient data input -> real-time XGBoost prediction with SHAP waterfall
2. LLM-generated clinical report (with template fallback)
3. RAG-powered clinical Q&A over medical literature

Usage: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os

st.set_page_config(
    page_title="Thyroid Disease Classifier",
    page_icon="🧬",
    layout="wide"
)

# Reference ranges for clinical interpretation
REF_RANGES = {
    "TSH": (0.4, 4.0, "mIU/L"),
    "T3": (0.8, 2.0, "ng/dL"),
    "T4": (60, 120, "nmol/L"),
    "T4U": (0.7, 1.2, ""),
    "FTI": (60, 120, ""),
}

PRESET_PATIENTS = {
    "Custom": {},
    "Hypothyroid Patient": {
        "TSH": 12.0, "T3": 0.8, "T4": 60.0, "T4U": 1.0,
        "age": 55, "sex": 0, "on_thyroxine": 1, "on_antithyroid": 0,
        "sick": 0, "pregnant": 0, "thyroid_surgery": 0,
    },
    "Hyperthyroid Patient": {
        "TSH": 0.1, "T3": 5.2, "T4": 180.0, "T4U": 1.3,
        "age": 35, "sex": 0, "on_thyroxine": 0, "on_antithyroid": 1,
        "sick": 0, "pregnant": 0, "thyroid_surgery": 0,
    },
    "Healthy Patient": {
        "TSH": 2.5, "T3": 1.8, "T4": 105.0, "T4U": 1.0,
        "age": 30, "sex": 1, "on_thyroxine": 0, "on_antithyroid": 0,
        "sick": 0, "pregnant": 0, "thyroid_surgery": 0,
    },
    "Borderline Subclinical": {
        "TSH": 5.5, "T3": 1.5, "T4": 85.0, "T4U": 0.9,
        "age": 62, "sex": 0, "on_thyroxine": 0, "on_antithyroid": 0,
        "sick": 0, "pregnant": 0, "thyroid_surgery": 0,
    },
}


@st.cache_resource
def load_model():
    models_dir = "models"
    meta_files = sorted(
        [f for f in os.listdir(models_dir) if f.startswith("metadata_")],
        reverse=True
    )
    if not meta_files:
        st.error("No trained model found. Run `python train.py` first.")
        st.stop()

    timestamp = meta_files[0].replace("metadata_", "").replace(".json", "")
    model = joblib.load(f"{models_dir}/best_model_{timestamp}.pkl")
    scaler = joblib.load(f"{models_dir}/scaler_{timestamp}.pkl")
    label_encoder = joblib.load(f"{models_dir}/label_encoder_{timestamp}.pkl")

    with open(f"{models_dir}/metadata_{timestamp}.json") as f:
        metadata = json.load(f)

    # Extract XGBoost sub-model for SHAP (TreeExplainer can't handle VotingClassifier)
    xgb_model = None
    if hasattr(model, 'named_estimators_'):
        for name, est in model.named_estimators_.items():
            if 'XGB' in type(est).__name__ or 'xgb' in name.lower():
                xgb_model = est
                break
    if xgb_model is None:
        xgb_model = model

    return model, xgb_model, scaler, label_encoder, metadata


@st.cache_resource
def load_rag():
    try:
        from rag.retriever import ThyroidRetriever
        return ThyroidRetriever()
    except Exception:
        return None


@st.cache_resource
def load_report_generator():
    try:
        from llm.report_generator import ClinicalReportGenerator
        return ClinicalReportGenerator()
    except Exception:
        return None


@st.cache_resource
def load_qa():
    try:
        from llm.clinical_qa import ClinicalQA
        return ClinicalQA()
    except Exception:
        return None


def get_shap_values(xgb_model, input_scaled, features, prediction_idx):
    """Safely compute SHAP values, handling all edge cases."""
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(input_scaled)
        idx = int(prediction_idx)

        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[idx]).flatten()
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                sv = shap_vals[0, :, idx]
            elif shap_vals.ndim == 2:
                sv = shap_vals[0]
            else:
                sv = shap_vals.flatten()
        else:
            return None

        if len(sv) == len(features):
            return dict(zip(features, [float(v) for v in sv]))
        return None
    except Exception:
        return None


def analyze_patient_data(patient):
    """Generate clinical findings from raw patient data vs reference ranges."""
    findings = []
    for marker, (low, high, unit) in REF_RANGES.items():
        val = patient.get(marker)
        if val is None:
            continue
        unit_str = f" {unit}" if unit else ""
        if val > high:
            findings.append((marker, val, f"ELEVATED ({val:.2f}{unit_str}, normal: {low}-{high})", True))
        elif val < low:
            findings.append((marker, val, f"LOW ({val:.2f}{unit_str}, normal: {low}-{high})", True))
        else:
            findings.append((marker, val, f"Normal ({val:.2f}{unit_str}, range: {low}-{high})", False))
    findings.sort(key=lambda x: 0 if x[3] else 1)
    return findings


model, xgb_model, scaler, label_encoder, metadata = load_model()
features = metadata["features_selected"]
if isinstance(features, int):
    features = [f"feature_{i}" for i in range(features)]

# Sidebar
st.sidebar.title("🧬 Thyroid Classifier")
st.sidebar.markdown("**XGBoost + SHAP + RAG + LLM**")
st.sidebar.markdown(f"Model: {metadata['best_model']}")
best_metrics = metadata["performance_metrics"][metadata["best_model"]]
st.sidebar.markdown(f"Accuracy: {best_metrics['accuracy']:.1%}")
st.sidebar.markdown(f"Features: {len(features)}")
st.sidebar.markdown("---")
rag_sys = load_rag()
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
st.sidebar.markdown(f"RAG: {'✅ Online' if rag_sys else '❌ Offline'}")
st.sidebar.markdown(f"LLM: {'✅ API Key Set' if api_key else '⚡ Template Mode'}")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", ["🔬 Predict & Explain", "📚 Clinical Q&A", "📊 Model Info"])


if page == "🔬 Predict & Explain":
    st.title("🔬 Thyroid Disease Prediction")
    st.markdown("Enter patient lab values to get a prediction with SHAP explanation and clinical report.")

    preset = st.selectbox("Load preset patient:", list(PRESET_PATIENTS.keys()))
    pv = PRESET_PATIENTS[preset]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Lab Values")
        patient = {}
        patient["TSH"] = st.slider("TSH (mIU/L)", 0.01, 50.0, pv.get("TSH", 2.5), 0.1,
                                   help="Normal: 0.4-4.0 mIU/L")
        patient["T3"] = st.slider("T3 (ng/dL)", 0.4, 6.5, pv.get("T3", 1.8), 0.1,
                                  help="Normal: 0.8-2.0 ng/dL")
        patient["T4"] = st.slider("T4 (nmol/L)", 35.0, 230.0, pv.get("T4", 105.0), 1.0,
                                  help="Normal: 60-120 nmol/L")
        patient["T4U"] = st.slider("T4 Uptake", 0.5, 1.8, pv.get("T4U", 1.0), 0.05)
        patient["FTI"] = patient["T4"] / (patient["T4U"] + 0.01)

        st.subheader("Demographics & History")
        patient["age"] = st.slider("Age", 18, 90, pv.get("age", 45))
        patient["sex"] = st.selectbox("Sex", [0, 1], index=pv.get("sex", 0),
                                      format_func=lambda x: "Female" if x == 0 else "Male")
        patient["on_thyroxine"] = st.selectbox("On Thyroxine?", [0, 1],
                                                index=pv.get("on_thyroxine", 0),
                                                format_func=lambda x: "No" if x == 0 else "Yes")
        patient["on_antithyroid"] = st.selectbox("On Antithyroid Medication?", [0, 1],
                                                  index=pv.get("on_antithyroid", 0),
                                                  format_func=lambda x: "No" if x == 0 else "Yes")
        patient["sick"] = st.selectbox("Currently Sick?", [0, 1],
                                       index=pv.get("sick", 0),
                                       format_func=lambda x: "No" if x == 0 else "Yes")
        patient["pregnant"] = st.selectbox("Pregnant?", [0, 1],
                                           index=pv.get("pregnant", 0),
                                           format_func=lambda x: "No" if x == 0 else "Yes")
        patient["thyroid_surgery"] = st.selectbox("Previous Thyroid Surgery?", [0, 1],
                                                   index=pv.get("thyroid_surgery", 0),
                                                   format_func=lambda x: "No" if x == 0 else "Yes")

    for f in features:
        if f not in patient:
            patient[f] = 0

    if st.button("🔍 Predict & Generate Report", type="primary"):
        input_df = pd.DataFrame([{f: patient.get(f, 0) for f in features}])
        input_scaled = scaler.transform(input_df)

        prediction_idx = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        prediction = label_encoder.inverse_transform([int(prediction_idx)])[0]
        confidence = float(probabilities[int(prediction_idx)])

        with col2:
            color_map = {"negative": "green", "hypothyroid": "orange", "hyperthyroid": "red"}
            st.subheader("Prediction Result")
            st.markdown(f"### :{color_map.get(prediction, 'blue')}[{prediction.upper()}]")
            st.markdown(f"**Confidence: {confidence:.1%}**")

            st.subheader("Class Probabilities")
            for cls, prob in zip(label_encoder.classes_, probabilities):
                st.progress(float(prob), text=f"{cls}: {float(prob):.1%}")

            # SHAP
            st.subheader("Feature Contributions")
            shap_dict = get_shap_values(xgb_model, input_scaled, features, prediction_idx)

            if shap_dict and any(abs(v) > 0.001 for v in shap_dict.values()):
                sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
                shap_df = pd.DataFrame(sorted_shap[:8], columns=["Feature", "SHAP Value"])
                st.bar_chart(shap_df.set_index("Feature"))
            else:
                st.markdown("*Clinical reference range analysis:*")
                findings = analyze_patient_data(patient)
                for marker, val, desc, abnormal in findings[:6]:
                    st.markdown(f"{'⚠️' if abnormal else '✅'} **{marker}**: {desc}")
                shap_dict = None

            # Clinical Report
            st.subheader("📋 Clinical Report")
            generator = load_report_generator()

            rag_context = ""
            if rag_sys:
                rag_context = rag_sys.get_context(
                    f"thyroid {prediction} TSH {patient.get('TSH', '')} diagnosis",
                    n_results=3
                )

            if generator:
                report = generator.generate(
                    patient_data=patient,
                    prediction=prediction,
                    confidence=confidence,
                    shap_values=shap_dict if shap_dict else {f: 0.0 for f in features},
                    rag_context=rag_context,
                )
                if "[LLM unavailable" in report:
                    report = generator.generate_from_data(patient, prediction, confidence, REF_RANGES)
                st.markdown(report)
            else:
                st.info("Report generator unavailable.")


elif page == "📚 Clinical Q&A":
    st.title("📚 Clinical Q&A — RAG-Powered")
    st.markdown("Ask questions about thyroid disease. Answers are grounded in medical literature.")

    examples = [
        "What causes elevated TSH levels?",
        "How does SMOTE improve thyroid disease classification?",
        "What is the role of SHAP in clinical ML models?",
        "What are the treatment options for hyperthyroidism?",
        "How does pregnancy affect thyroid function?",
    ]

    selected_example = st.selectbox("Example questions:", [""] + examples)
    question = st.text_input("Your question:", value=selected_example)

    if question:
        qa = load_qa()
        if qa:
            with st.spinner("Searching medical literature..."):
                result = qa.ask(question)
            st.subheader("Answer")
            st.markdown(result["answer"])
            st.subheader("Sources")
            for src in result["sources"]:
                st.markdown(f"- **{src['title']}** — *{src['source']}*")
        else:
            if rag_sys:
                docs = rag_sys.search(question)
                st.subheader("Retrieved Literature")
                for doc in docs:
                    with st.expander(f"📄 {doc['title']}"):
                        st.markdown(f"*{doc['source']}*")
                        st.markdown(doc["text"])
            else:
                st.error("RAG system unavailable. Install chromadb: `pip install chromadb`")


elif page == "📊 Model Info":
    st.title("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    best = metadata["best_model"]
    metrics = metadata["performance_metrics"][best]
    col1.metric("Best Model", best)
    col2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col3.metric("F1 Score", f"{metrics['f1_score']:.1%}")

    st.subheader("All Models")
    perf_df = pd.DataFrame(metadata["performance_metrics"]).T
    st.dataframe(perf_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Selected Features")
    st.markdown(", ".join(f"`{f}`" for f in features))

    st.subheader("Dataset")
    st.markdown(f"- **Samples:** {metadata['dataset_size']:,}")
    st.markdown(f"- **Features selected:** {len(features)}")
    st.markdown(f"- **Target classes:** {', '.join(metadata['target_classes'])}")

    st.subheader("Architecture")
    st.code("""
Patient Data -> Feature Selection (RFE: 19->12) -> SMOTE Balancing
    -> XGBoost + Random Forest + Ensemble
    -> SHAP Explainability
    -> RAG (ChromaDB + 25 Medical Documents)
    -> LLM Clinical Report Generation
    -> Streamlit Dashboard
    """, language="text")

    st.subheader("RAG Knowledge Base")
    try:
        from rag.documents import THYROID_LITERATURE
        st.markdown(f"**{len(THYROID_LITERATURE)} indexed medical documents** covering:")
        topics = set()
        for doc in THYROID_LITERATURE:
            for kw in ["diagnosis", "treatment", "ML", "SHAP", "RAG", "pregnancy", "autoimmune", "subclinical"]:
                if kw.lower() in doc["title"].lower() or kw.lower() in doc["text"].lower():
                    topics.add(kw)
        st.markdown(", ".join(sorted(topics)))
    except Exception:
        st.info("RAG system not initialized")

    plots_dir = "outputs/plots"
    if os.path.exists(plots_dir):
        st.subheader("Training Visualizations")
        images = sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])
        if images:
            cols = st.columns(2)
            for i, img in enumerate(images[:4]):
                with cols[i % 2]:
                    st.image(f"{plots_dir}/{img}",
                             caption=img.replace("_", " ").replace(".png", "").title())
