"""
Generate clinical reports from ML predictions using LLM.

Takes XGBoost prediction + SHAP values + patient data
and generates a natural language clinical summary.
Falls back to data-driven template when no LLM API is available.
"""
import os
import json
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

LLM_PROVIDER = None
try:
    import anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        LLM_PROVIDER = "anthropic"
except ImportError:
    pass

if not LLM_PROVIDER:
    try:
        import openai
        if os.environ.get("OPENAI_API_KEY"):
            LLM_PROVIDER = "openai"
    except ImportError:
        pass


class ClinicalReportGenerator:
    """Generate natural language clinical reports from ML predictions."""

    def __init__(self):
        self.provider = LLM_PROVIDER
        if self.provider == "anthropic":
            import anthropic as _a
            self.client = _a.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        elif self.provider == "openai":
            import openai as _o
            self.client = _o.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        else:
            self.client = None

    def generate(self, patient_data, prediction, confidence, shap_values, rag_context=""):
        """Generate report using LLM if available, otherwise template."""
        # Check if SHAP values are meaningful
        has_shap = shap_values and any(abs(v) > 0.001 for v in shap_values.values())

        if self.provider and self.client:
            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            prompt = self._build_prompt(patient_data, prediction, confidence, sorted_shap, rag_context)
            try:
                if self.provider == "anthropic":
                    return self._call_anthropic(prompt)
                elif self.provider == "openai":
                    return self._call_openai(prompt)
            except Exception as e:
                log.warning(f"LLM call failed: {e}")

        # Fallback: data-driven template
        ref_ranges = {
            "TSH": (0.4, 4.0, "mIU/L"),
            "T3": (0.8, 2.0, "ng/dL"),
            "T4": (60, 120, "nmol/L"),
            "T4U": (0.7, 1.2, ""),
            "FTI": (60, 120, ""),
        }

        if has_shap:
            return self._generate_shap_template(patient_data, prediction, confidence, shap_values, ref_ranges)
        else:
            return self.generate_from_data(patient_data, prediction, confidence, ref_ranges)

    def generate_from_data(self, patient_data, prediction, confidence, ref_ranges):
        """Generate report from patient data analysis (no SHAP needed)."""
        tsh = patient_data.get("TSH", 0)
        t3 = patient_data.get("T3", 0)
        t4 = patient_data.get("T4", 0)
        age = patient_data.get("age", 0)

        # Analyze values against reference ranges
        findings = []
        for marker, (low, high, unit) in ref_ranges.items():
            val = patient_data.get(marker)
            if val is None:
                continue
            unit_str = f" {unit}" if unit else ""
            if val > high:
                findings.append(f"**{marker}**: {val:.2f}{unit_str} — **ELEVATED** (normal: {low}-{high})")
            elif val < low:
                findings.append(f"**{marker}**: {val:.2f}{unit_str} — **LOW** (normal: {low}-{high})")
            else:
                findings.append(f"**{marker}**: {val:.2f}{unit_str} — within normal range ({low}-{high})")

        # Clinical interpretation based on prediction
        interpretation = {
            "hypothyroid": (
                f"elevated TSH ({tsh:.2f} mIU/L) with {'low' if t4 < 60 else 'borderline'} T4 "
                f"levels ({t4:.1f} nmol/L), consistent with primary hypothyroidism"
            ),
            "hyperthyroid": (
                f"suppressed TSH ({tsh:.2f} mIU/L) with {'elevated' if t3 > 2.0 else 'borderline'} T3 "
                f"({t3:.2f} ng/dL) and {'elevated' if t4 > 120 else 'borderline'} T4 ({t4:.1f} nmol/L), "
                f"suggesting hyperthyroidism"
            ),
            "negative": (
                f"thyroid markers within or near normal reference ranges "
                f"(TSH: {tsh:.2f}, T3: {t3:.2f}, T4: {t4:.1f}), suggesting euthyroid state"
            ),
        }

        next_steps = {
            "hypothyroid": [
                "Repeat TSH and free T4 in 6-8 weeks to confirm",
                "Consider anti-TPO antibody testing to assess autoimmune etiology",
                "If confirmed, initiate levothyroxine therapy per clinical guidelines",
            ],
            "hyperthyroid": [
                "Order TSH receptor antibodies (TRAb) to evaluate for Graves disease",
                "Consider radioactive iodine uptake scan",
                "Assess for symptoms: tremor, weight loss, heat intolerance, palpitations",
            ],
            "negative": [
                "No immediate thyroid intervention required",
                "Routine screening per age/sex-appropriate guidelines",
                "Re-evaluate if symptoms develop",
            ],
        }

        interp = interpretation.get(prediction, "thyroid function assessment pending clinical correlation")
        steps = next_steps.get(prediction, ["Correlate with clinical presentation"])

        report = f"""**CLINICAL REPORT — Thyroid Function Assessment**

**1. CLINICAL SUMMARY**

The ML model predicts **{prediction.upper()}** with **{confidence:.1%}** confidence, based on {interp}.

**2. LAB VALUE ANALYSIS**

"""
        for f in findings:
            report += f"- {f}\n"

        report += f"""
**3. CLINICAL CONTEXT**

{'The patient is ' + str(int(age)) + ' years old.' if age else ''} """

        if patient_data.get("on_thyroxine"):
            report += "Currently on levothyroxine therapy. "
        if patient_data.get("on_antithyroid"):
            report += "Currently on antithyroid medication. "
        if patient_data.get("pregnant"):
            report += "Patient is pregnant — trimester-specific TSH ranges apply. "

        report += f"""

**4. RECOMMENDED NEXT STEPS**

"""
        for step in steps:
            report += f"- {step}\n"

        report += "\n*Report generated by ML clinical decision support system. Not a substitute for clinical judgment.*"
        return report

    def _generate_shap_template(self, patient_data, prediction, confidence, shap_values, ref_ranges):
        """Generate report using SHAP values."""
        tsh = patient_data.get("TSH", 0)
        t3 = patient_data.get("T3", 0)
        t4 = patient_data.get("T4", 0)

        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_shap[:3]

        report = f"""**CLINICAL REPORT — Thyroid Function Assessment**

**1. CLINICAL SUMMARY**

The ML model predicts **{prediction.upper()}** with **{confidence:.1%}** confidence.
Key lab values: TSH={tsh:.2f} mIU/L, T3={t3:.2f} ng/dL, T4={t4:.1f} nmol/L.

**2. KEY CONTRIBUTING FEATURES (SHAP Analysis)**

"""
        for name, value in top3:
            direction = "pushes toward" if value > 0 else "pushes away from"
            strength = "strongly" if abs(value) > 1.0 else "moderately" if abs(value) > 0.3 else "slightly"
            actual_val = patient_data.get(name, "N/A")
            ref = ref_ranges.get(name)
            ref_str = f" (normal: {ref[0]}-{ref[1]} {ref[2]})" if ref else ""
            report += f"- **{name}** = {actual_val}{ref_str}: {strength} {direction} {prediction} (SHAP: {value:+.3f})\n"

        report += f"""
**3. MODEL EXPLANATION**

The prediction is primarily driven by {top3[0][0]} (contribution: {top3[0][1]:+.3f}), followed by {top3[1][0]} ({top3[1][1]:+.3f}) and {top3[2][0]} ({top3[2][1]:+.3f}).

**4. RECOMMENDED NEXT STEPS**

- Correlate ML findings with clinical presentation and patient history
- Consider repeat thyroid function tests in 6-8 weeks if subclinical
- Refer to endocrinology if overt thyroid dysfunction confirmed

*Report generated by ML clinical decision support system. Not a substitute for clinical judgment.*"""
        return report

    def _build_prompt(self, patient_data, prediction, confidence, top_features, rag_context):
        feature_lines = "\n".join(
            f"- {name}: contribution={value:+.3f}" for name, value in top_features
        )
        lit = f"\nRELEVANT MEDICAL LITERATURE:\n{rag_context}" if rag_context else ""

        return f"""You are a clinical decision support system for thyroid disease diagnosis.
Generate a concise clinical report based on the following ML model prediction and patient data.

PATIENT DATA:
{json.dumps(patient_data, indent=2)}

ML PREDICTION:
- Diagnosis: {prediction}
- Confidence: {confidence:.1%}

TOP CONTRIBUTING FEATURES (SHAP Analysis):
{feature_lines}

REFERENCE RANGES:
- TSH: 0.4-4.0 mIU/L
- T3: 0.8-2.0 ng/dL
- T4: 60-120 nmol/L
{lit}

Generate a clinical report with these sections:
1. CLINICAL SUMMARY (2-3 sentences)
2. KEY FINDINGS (top 3 features with reference ranges)
3. SHAP EXPLANATION (plain English)
4. CLINICAL CONTEXT (from literature if provided)
5. RECOMMENDED NEXT STEPS (2-3 bullet points)

Keep under 250 words. Use clinical language understandable to a general practitioner."""

    def _call_anthropic(self, prompt):
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _call_openai(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
