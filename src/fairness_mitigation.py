"""
Equalized-recall post-processing (Hardt, Price & Srebro 2016-style):
instead of one global risk-score threshold, calibrate a PER-AGE-GROUP
threshold so every age bucket catches the same fraction of its true
readmissions (recall), directly targeting the gap found in the
fairness audit (0-30 group: 26.7% recall vs 38-42% for other groups
under a single global threshold).

Important tradeoff, measured explicitly below (not glossed over):
lowering the threshold for an under-served group necessarily flags MORE
of that group's patients to catch the same recall -- selection rate
goes up and precision goes down for that group. This is inherent to the
fairness-metric tradeoff (Kleinberg et al. 2016), not a bug in the fix.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve

from fairness_audit import group_metrics, MIN_GROUP_SIZE

TARGET = "readmit_30"


def threshold_for_target_recall(y_true, scores, target_recall):
    """Smallest threshold whose recall is >= target_recall (i.e. the
    LEAST aggressive threshold that still hits the target -- avoids
    over-flagging beyond what's needed)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = np.where(tpr >= target_recall)[0]
    if len(valid) == 0:
        return scores.min()  # can't reach target even at minimum threshold
    # thresholds from roc_curve are descending; take the LAST index that
    # still clears target_recall -> highest threshold meeting the bar.
    idx = valid[0]
    return thresholds[idx]


def main():
    model = joblib.load("../results/xgb_model.pkl")
    preprocessor = joblib.load("../results/preprocessor.pkl")
    test_df = pd.read_csv("../data/test.csv")

    y_true = test_df[TARGET].values
    X_test = preprocessor.transform(test_df.drop(columns=[TARGET]))
    scores = model.predict_proba(X_test)[:, 1]

    age_bucket = pd.cut(
        test_df["age_ordinal"], bins=[-1, 2, 5, 7, 9],
        labels=["0-30", "30-60", "60-80", "80-100"]
    ).astype(str)

    # --- BEFORE: single global threshold (top 20%) ---
    k = int(len(scores) * 0.20)
    global_threshold = np.sort(scores)[::-1][k - 1]
    flagged_before = (scores >= global_threshold).astype(int)

    # --- Target: overall recall the global policy achieved ---
    overall_recall_before = ((flagged_before == 1) & (y_true == 1)).sum() / y_true.sum()
    print(f"Global-threshold policy: overall recall = {overall_recall_before:.3f}, "
          f"threshold = {global_threshold:.3f}\n")
    print(f"Calibrating per-age-group thresholds to each hit recall >= "
          f"{overall_recall_before:.3f}...\n")

    flagged_after = np.zeros_like(flagged_before)
    group_thresholds = {}
    for group in sorted(age_bucket.unique()):
        mask = (age_bucket == group).values
        y_g, scores_g = y_true[mask], scores[mask]
        if y_g.sum() < 10:
            # too few positives to calibrate reliably -- fall back to global
            thresh = global_threshold
        else:
            thresh = threshold_for_target_recall(y_g, scores_g, overall_recall_before)
        group_thresholds[group] = thresh
        flagged_after[mask] = (scores_g >= thresh).astype(int)
        print(f"  {group:10s}: threshold {global_threshold:.3f} -> {thresh:.3f}")

    print(f"\nTotal flagged -- before: {flagged_before.sum()}, after: {flagged_after.sum()} "
          f"(out of {len(scores)} patients)")

    print("\n" + "=" * 100)
    print("BEFORE (single global threshold) vs AFTER (per-age-group calibrated threshold)")
    print("=" * 100)
    rows = []
    for group in sorted(age_bucket.unique()):
        mask = (age_bucket == group).values
        before = group_metrics(y_true, scores, flagged_before, group, mask)
        after = group_metrics(y_true, scores, flagged_after, group, mask)
        rows.append({
            "group": group, "n": before["n"], "n_positives": before["n_positives"],
            "recall_before": before["recall_at_top20"], "recall_after": after["recall_at_top20"],
            "precision_before": before["precision_at_top20"], "precision_after": after["precision_at_top20"],
            "selection_rate_before": before["selection_rate"], "selection_rate_after": after["selection_rate"],
        })
    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    recall_gap_before = result_df["recall_before"].max() - result_df["recall_before"].min()
    recall_gap_after = result_df["recall_after"].max() - result_df["recall_after"].min()
    overall_precision_before = ((flagged_before == 1) & (y_true == 1)).sum() / flagged_before.sum()
    overall_precision_after = ((flagged_after == 1) & (y_true == 1)).sum() / flagged_after.sum()

    print(f"\nRecall gap across age groups: {recall_gap_before:.3f} -> {recall_gap_after:.3f}")
    print(f"Overall precision (all patients): {overall_precision_before:.3f} -> "
          f"{overall_precision_after:.3f}")
    print(f"Overall patients flagged: {flagged_before.sum()} -> {flagged_after.sum()} "
          f"({100*(flagged_after.sum()/flagged_before.sum()-1):+.1f}%)")

    print("""
INTERPRETATION: recall gap should shrink substantially -- that's the
fix working. The cost: total patients flagged (and therefore overall
precision) likely rises, because equalizing the 0-30 group's recall
requires flagging more of that group, without un-flagging anyone else
under this simple version. A hospital with a hard intervention-capacity
budget would need to combine this with a global cap (e.g. re-optimize
group thresholds jointly to hit a fixed total flagged count) -- noted
as a real next step, not implemented here to keep the demonstration
readable.
""")

    result_df.to_csv("../results/fairness_mitigation_before_after.csv", index=False)
    print("Saved ../results/fairness_mitigation_before_after.csv")


if __name__ == "__main__":
    main()
