"""
SHAP (SHapley Additive exPlanations) explainability for the XGBoost
readmission model.

The core idea, briefly: for a single prediction, SHAP answers "how much
did each feature push this specific patient's risk score up or down,
relative to the average patient?" It's grounded in cooperative game
theory (Shapley values) -- the contributions are guaranteed to sum
exactly to (this patient's prediction - the average prediction), which
is what makes it trustworthy rather than a heuristic feature-importance
hack.

Two views, both needed for a clinician-facing tool:
  - GLOBAL: which features matter most across ALL patients (model-level
    trust -- "does this model rely on things a doctor would expect?")
  - LOCAL: for ONE patient, exactly why the model flagged them (this is
    what actually gets shown at the bedside)
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # no display needed, we're saving PNGs
import matplotlib.pyplot as plt

TARGET = "readmit_30"


def main():
    model = joblib.load("../results/xgb_model.pkl")
    preprocessor = joblib.load("../results/preprocessor.pkl")
    feature_names = np.load("../results/X_test_columns.npy", allow_pickle=True)

    test_df = pd.read_csv("../data/test.csv")
    y_test = test_df[TARGET].values
    X_test = preprocessor.transform(test_df.drop(columns=[TARGET]))
    X_test_dense = np.asarray(X_test.todense()) if hasattr(X_test, "todense") else X_test

    print("Computing SHAP values (TreeExplainer -- exact & fast for XGBoost)...")
    explainer = shap.TreeExplainer(model)
    # Use a subsample for speed -- SHAP values across 20k rows x 230
    # features is slow on a laptop CPU; 2000 rows is plenty for a
    # trustworthy global-importance picture.
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(X_test_dense), size=min(2000, len(X_test_dense)), replace=False)
    X_sample = X_test_dense[sample_idx]
    shap_values = explainer.shap_values(X_sample)

    # --- GLOBAL: top features driving readmission risk, across patients ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]
    print("\n=== Top 15 features by global importance (mean |SHAP value|) ===")
    for i in top_idx:
        print(f"  {feature_names[i]:40s} {mean_abs_shap[i]:.4f}")

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                       max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("../results/shap_global_summary.png", dpi=150)
    plt.close()
    print("\nSaved ../results/shap_global_summary.png")

    # --- LOCAL: explain one specific high-risk patient ---
    full_scores = model.predict_proba(X_test_dense)[:, 1]
    # Pick the highest-risk patient who actually WAS readmitted (a true
    # positive) -- the most useful case to show: "the model caught this
    # one, and here's why."
    candidates = np.where(y_test == 1)[0]
    patient_idx = candidates[np.argmax(full_scores[candidates])]

    explainer_full = shap.TreeExplainer(model)
    patient_shap = explainer_full.shap_values(X_test_dense[patient_idx:patient_idx + 1])[0]
    base_value = explainer_full.expected_value

    print(f"\n=== Local explanation: patient #{patient_idx} "
          f"(predicted risk={full_scores[patient_idx]:.1%}, actually readmitted=True) ===")
    print(f"Base rate (average patient): {1 / (1 + np.exp(-base_value)):.1%}\n")
    order = np.argsort(np.abs(patient_shap))[::-1][:10]
    print(f"{'Feature':40s} {'Value pushed risk':>18s}")
    for i in order:
        direction = "UP" if patient_shap[i] > 0 else "down"
        print(f"  {feature_names[i]:38s} {direction:>5s} by {abs(patient_shap[i]):.4f}")

    plt.figure()
    shap.force_plot(base_value, patient_shap, X_test_dense[patient_idx],
                     feature_names=feature_names, matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig("../results/shap_local_patient_example.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved ../results/shap_local_patient_example.png")


if __name__ == "__main__":
    main()
