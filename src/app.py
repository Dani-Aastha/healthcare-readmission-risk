"""
Streamlit demo: clinician-facing readmission-risk view.

Run with:  streamlit run app.py   (from the src/ directory, venv active)

Three panels:
  1. Patient picker + risk score
  2. SHAP-based "why" explanation, in plain language
  3. Fairness before/after toggle (the age-threshold fix)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve paths relative to THIS file, not the process's working
# directory -- the dev-server launcher may invoke `streamlit run` from
# a different cwd than `src/`, and relative "../results/..." paths
# would then silently point at the wrong place (or just fail).
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))  # so `from id_mappings import ...` works regardless of cwd

from id_mappings import load_id_mappings

TARGET = "readmit_30"

st.set_page_config(page_title="Readmission Risk Explainer", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(RESULTS_DIR / "xgb_model.pkl")
    preprocessor = joblib.load(RESULTS_DIR / "preprocessor.pkl")
    feature_names = np.load(RESULTS_DIR / "X_test_columns.npy", allow_pickle=True)
    explainer = shap.TreeExplainer(model)
    id_maps = load_id_mappings(str(DATA_DIR / "IDS_mapping.csv"))
    return model, preprocessor, feature_names, explainer, id_maps


@st.cache_data
def load_data():
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return test_df


def readable_feature_name(raw_name: str) -> str:
    """cat__medical_specialty_Missing -> 'Medical specialty: Missing'
    num__number_inpatient -> 'Number inpatient'"""
    name = raw_name.split("__", 1)[-1]
    LABELS = {
        "number_inpatient": "Prior inpatient visits (past year)",
        "number_emergency": "Prior emergency visits (past year)",
        "number_outpatient": "Prior outpatient visits (past year)",
        "discharge_disposition_id": "Discharge disposition (numeric code)",
        "admission_type_id": "Admission type (numeric code)",
        "admission_source_id": "Admission source (numeric code)",
        "time_in_hospital": "Length of stay (days)",
        "num_medications": "Number of medications",
        "num_lab_procedures": "Number of lab procedures",
        "num_procedures": "Number of procedures",
        "number_diagnoses": "Number of diagnoses",
        "age_ordinal": "Age group",
    }
    if name in LABELS:
        return LABELS[name]
    # e.g. "medical_specialty_Cardiology" -> "Medical specialty: Cardiology"
    for prefix, label in [("medical_specialty_", "Medical specialty: "),
                            ("payer_code_", "Payer code: "),
                            ("race_", "Race: "),
                            ("diag_1_", "Primary diagnosis: "),
                            ("diag_2_", "Secondary diagnosis: "),
                            ("diag_3_", "Additional diagnosis: "),
                            ("age_", "Age bracket: "),
                            ("insulin_", "Insulin: "),
                            ("diabetesMed_", "On diabetes medication: ")]:
        if name.startswith(prefix):
            return label + name[len(prefix):]
    return name.replace("_", " ").capitalize()


def main():
    model, preprocessor, feature_names, explainer, id_maps = load_artifacts()
    test_df = load_data()

    st.title("🏥 30-Day Readmission Risk — Explainer Demo")
    st.caption("Trained on the UCI Diabetes 130-US Hospitals dataset (101,766 real encounters). "
               "Not a clinical tool — a portfolio demonstration of an explainable risk model.")

    tab1, tab2 = st.tabs(["Patient Risk Explanation", "Fairness: Before / After Fix"])

    # ---------------- TAB 1: Patient explanation ----------------
    with tab1:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Select a patient")
            filter_choice = st.radio("Show:", ["Any patient", "Actually readmitted (<30d)",
                                                  "Not readmitted"])
            pool = test_df
            if filter_choice == "Actually readmitted (<30d)":
                pool = test_df[test_df[TARGET] == 1]
            elif filter_choice == "Not readmitted":
                pool = test_df[test_df[TARGET] == 0]

            if "patient_idx" not in st.session_state or st.button("🔀 Pick random patient"):
                st.session_state.patient_idx = pool.sample(1, random_state=None).index[0]
            patient_idx = st.session_state.patient_idx
            if patient_idx not in pool.index:
                patient_idx = pool.index[0]
                st.session_state.patient_idx = patient_idx

            patient_row = test_df.loc[[patient_idx]]

            st.markdown("**Patient snapshot**")
            snap = {
                "Age": patient_row["age"].values[0],
                "Race": patient_row["race"].values[0],
                "Gender": patient_row["gender"].values[0],
                "Time in hospital (days)": int(patient_row["time_in_hospital"].values[0]),
                "Prior inpatient visits": int(patient_row["number_inpatient"].values[0]),
                "Discharge disposition": id_maps.get("discharge_disposition_id", {}).get(
                    int(patient_row["discharge_disposition_id"].values[0]), "Unknown"),
                "Actually readmitted <30d": "Yes" if patient_row[TARGET].values[0] == 1 else "No",
            }
            st.table(pd.DataFrame(snap.items(), columns=["Field", "Value"]).set_index("Field"))

        with col_b:
            X_patient = preprocessor.transform(patient_row.drop(columns=[TARGET]))
            X_patient_dense = np.asarray(X_patient.todense()) if hasattr(X_patient, "todense") else X_patient
            risk_score = model.predict_proba(X_patient_dense)[0, 1]

            risk_label = "🔴 HIGH RISK" if risk_score >= 0.573 else "🟢 Lower risk"
            st.metric("Predicted 30-day readmission risk", f"{risk_score:.1%}", risk_label)
            st.caption("Flagged 'high risk' if score >= 0.573 (the top-20%-of-population "
                       "threshold used hospital-wide).")

            shap_vals = explainer.shap_values(X_patient_dense)[0]
            base_value = explainer.expected_value

            order = np.argsort(np.abs(shap_vals))[::-1][:8]
            labels = [readable_feature_name(feature_names[i]) for i in order]
            values = [shap_vals[i] for i in order]
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

            fig, ax = plt.subplots(figsize=(7, 4))
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Contribution to risk score (SHAP value)")
            ax.set_title("Why this score: top contributing factors")
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("**In plain language:**")
            up_factors = [labels[i] for i, v in enumerate(values) if v > 0][:3]
            down_factors = [labels[i] for i, v in enumerate(values) if v < 0][:2]
            sentence = ""
            if up_factors:
                sentence += f"Risk is pushed **up** mainly by: {', '.join(up_factors)}. "
            if down_factors:
                sentence += f"Risk is pushed **down** by: {', '.join(down_factors)}."
            st.write(sentence)
            st.caption("🔴 red bars increase predicted risk · 🔵 blue bars decrease it — "
                       "bar length = size of that factor's contribution to THIS patient's score.")

    # ---------------- TAB 2: Fairness before/after ----------------
    with tab2:
        st.subheader("Age-group recall: before vs. after threshold calibration")
        st.write("A single global risk threshold under-served the youngest patient group "
                 "(0-30) — high AUROC (0.706, the model ranks them fine) but low recall under "
                 "one global cutoff, because their absolute scores run lower across the board. "
                 "Fixed with per-age-group threshold calibration (Hardt et al. 2016-style).")
        try:
            fair_df = pd.read_csv(RESULTS_DIR / "fairness_mitigation_before_after.csv")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            x = np.arange(len(fair_df))
            width = 0.35
            ax2.bar(x - width/2, fair_df["recall_before"], width, label="Before (global threshold)",
                    color="#d62728")
            ax2.bar(x + width/2, fair_df["recall_after"], width, label="After (per-group threshold)",
                    color="#2ca02c")
            ax2.set_xticks(x)
            ax2.set_xticklabels(fair_df["group"])
            ax2.set_ylabel("Recall (fraction of true readmissions caught)")
            ax2.set_title("Recall by age group: before vs after fix")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)

            gap_before = fair_df["recall_before"].max() - fair_df["recall_before"].min()
            gap_after = fair_df["recall_after"].max() - fair_df["recall_after"].min()
            c1, c2, c3 = st.columns(3)
            c1.metric("Recall gap before", f"{gap_before:.3f}")
            c2.metric("Recall gap after", f"{gap_after:.3f}",
                      delta=f"{gap_after - gap_before:+.3f}", delta_color="inverse")
            c3.metric("Overall precision change", "~unchanged",
                      help="0.213 -> 0.212 -- fixing this cost almost nothing")
        except FileNotFoundError:
            st.warning("Run fairness_mitigation.py first to generate this comparison.")


if __name__ == "__main__":
    main()
