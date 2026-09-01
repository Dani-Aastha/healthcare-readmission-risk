"""
Fairness audit across race, gender, and age.

Checks three DIFFERENT, NOT SIMULTANEOUSLY SATISFIABLE notions of
fairness (this is a real, proven impossibility result -- Kleinberg et
al. 2016 -- not a limitation of this analysis in particular):

  1. Selection rate parity ("demographic parity"): does every group get
     flagged high-risk at similar rates?
  2. Equal opportunity (recall/TPR parity): among patients who ACTUALLY
     get readmitted, does the model catch them at similar rates across
     groups? This is the clinically highest-stakes one -- a group with
     systematically lower recall means the model is quietly failing to
     protect those patients.
  3. Predictive parity (precision parity): among patients flagged
     high-risk, is the flag equally trustworthy across groups?

We use ONE global threshold (top 20% risk, same cutoff used in
train_model.py) rather than per-group thresholds, because that's how
this would actually be deployed -- a single risk score cutoff applied
hospital-wide, not a different bar for different demographic groups
(using per-group thresholds is itself a fairness intervention with its
own tradeoffs, not a neutral default).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

TARGET = "readmit_30"
MIN_GROUP_SIZE = 200          # below this, flag as low-confidence
MIN_POSITIVES_FOR_AUROC = 20  # AUROC is unstable with very few positives


def group_metrics(y_true, scores, flagged, group_name, group_mask):
    n = group_mask.sum()
    y_g = y_true[group_mask]
    scores_g = scores[group_mask]
    flagged_g = flagged[group_mask]

    prevalence = y_g.mean()
    selection_rate = flagged_g.mean()

    n_positives = y_g.sum()
    if n_positives >= MIN_POSITIVES_FOR_AUROC and len(set(y_g)) > 1:
        auroc = roc_auc_score(y_g, scores_g)
    else:
        auroc = np.nan

    true_positives = ((flagged_g == 1) & (y_g == 1)).sum()
    recall = true_positives / n_positives if n_positives > 0 else np.nan
    n_flagged = flagged_g.sum()
    precision = true_positives / n_flagged if n_flagged > 0 else np.nan

    return {
        "group": group_name,
        "n": n,
        "n_positives": n_positives,
        "low_confidence": n < MIN_GROUP_SIZE,
        "prevalence": prevalence,
        "selection_rate": selection_rate,
        "auroc": auroc,
        "recall_at_top20": recall,
        "precision_at_top20": precision,
    }


def audit(attr_name, attr_series, y_true, scores, flagged, min_group_size=MIN_GROUP_SIZE):
    rows = []
    for group_val in attr_series.unique():
        mask = (attr_series == group_val).values
        if mask.sum() < 30:  # too small to report at all
            continue
        rows.append(group_metrics(y_true, scores, flagged, str(group_val), mask))
    df = pd.DataFrame(rows).sort_values("n", ascending=False)

    print(f"\n{'='*90}\nFAIRNESS AUDIT: {attr_name}\n{'='*90}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    reliable = df[~df["low_confidence"]]
    if len(reliable) >= 2:
        recall_gap = reliable["recall_at_top20"].max() - reliable["recall_at_top20"].min()
        precision_gap = reliable["precision_at_top20"].max() - reliable["precision_at_top20"].min()
        selection_gap = reliable["selection_rate"].max() - reliable["selection_rate"].min()
        worst_recall_group = reliable.loc[reliable["recall_at_top20"].idxmin(), "group"]
        print(f"\n  Recall gap (among adequately-sized groups): {recall_gap:.3f}  "
              f"(lowest recall: {worst_recall_group})")
        print(f"  Precision gap: {precision_gap:.3f}")
        print(f"  Selection-rate gap: {selection_gap:.3f}")
        if recall_gap > 0.10:
            print(f"  >>> FLAG: >10pp recall gap -- '{worst_recall_group}' patients who will "
                  f"actually be readmitted are caught meaningfully less often. Investigate before "
                  f"deployment.")
    return df


def main():
    model = joblib.load("../results/xgb_model.pkl")
    preprocessor = joblib.load("../results/preprocessor.pkl")
    test_df = pd.read_csv("../data/test.csv")

    y_true = test_df[TARGET].values
    X_test = preprocessor.transform(test_df.drop(columns=[TARGET]))
    scores = model.predict_proba(X_test)[:, 1]

    k = int(len(scores) * 0.20)
    threshold = np.sort(scores)[::-1][k - 1]
    flagged = (scores >= threshold).astype(int)
    print(f"Global top-20% risk threshold: score >= {threshold:.3f} "
          f"({flagged.sum()} of {len(scores)} patients flagged)")

    race_df = audit("race", test_df["race"], y_true, scores, flagged)
    gender_df = audit("gender", test_df["gender"], y_true, scores, flagged)

    age_bucket = pd.cut(
        test_df["age_ordinal"], bins=[-1, 2, 5, 7, 9],
        labels=["0-30", "30-60", "60-80", "80-100"]
    )
    age_df = audit("age_bucket", age_bucket, y_true, scores, flagged)

    race_df.to_csv("../results/fairness_race.csv", index=False)
    gender_df.to_csv("../results/fairness_gender.csv", index=False)
    age_df.to_csv("../results/fairness_age.csv", index=False)
    print("\nSaved fairness_{race,gender,age}.csv to ../results/")


if __name__ == "__main__":
    main()
