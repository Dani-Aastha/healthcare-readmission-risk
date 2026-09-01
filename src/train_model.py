"""
Train and evaluate readmission-risk models.

Two models, deliberately:
  - Logistic Regression: interpretable-by-construction baseline. If a
    much fancier model can't beat this by a meaningful margin, that's
    important to know, not embarrassing to report.
  - XGBoost: the model we'll actually explain with SHAP.

Evaluation avoids the accuracy trap (see note above in chat) and adds a
metric a hospital administrator would actually recognize: precision/
recall if you can only act on your top-K% highest-risk patients.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

TARGET = "readmit_30"


def load_splits():
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    return train_df, test_df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    # NOTE: don't check `dtype == object` -- pandas 3.0 (released 2025)
    # gives string columns a dedicated `str` dtype instead of `object`,
    # so that check silently finds zero categorical columns on this
    # pandas version. is_numeric_dtype is robust to the backend.
    y_and_id_cols = {TARGET}
    categorical_cols = [c for c in df.columns
                        if not pd.api.types.is_numeric_dtype(df[c]) and c not in y_and_id_cols]
    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and c not in y_and_id_cols]
    print(f"  {len(categorical_cols)} categorical cols, {len(numeric_cols)} numeric cols")
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]), categorical_cols, numeric_cols


def precision_recall_at_k(y_true, scores, k_frac=0.20):
    """Of the top-k% highest-risk predictions, what fraction were true
    readmissions (precision), and what fraction of ALL true readmissions
    did we catch (recall)?"""
    n = len(scores)
    k = int(n * k_frac)
    top_k_idx = np.argsort(scores)[::-1][:k]
    caught = y_true[top_k_idx].sum()
    precision = caught / k
    recall = caught / y_true.sum()
    return precision, recall


def evaluate(name, y_true, scores):
    auroc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    prec20, rec20 = precision_recall_at_k(y_true, scores, 0.20)
    baseline_prevalence = y_true.mean()
    print(f"\n--- {name} ---")
    print(f"  AUROC:                 {auroc:.3f}  (0.5 = random, 1.0 = perfect)")
    print(f"  PR-AUC (avg precision): {ap:.3f}  (baseline/random = {baseline_prevalence:.3f})")
    print(f"  Top-20%-risk precision: {prec20:.3f}  "
          f"(vs {baseline_prevalence:.3f} if picking randomly)")
    print(f"  Top-20%-risk recall:    {rec20:.3f}  "
          f"(fraction of ALL readmissions caught by flagging top 20% risk)")
    return {"auroc": auroc, "pr_auc": ap, "precision@20": prec20, "recall@20": rec20}


def main():
    train_df, test_df = load_splits()
    y_train, y_test = train_df[TARGET].values, test_df[TARGET].values

    print("Building preprocessor...")
    preprocessor, cat_cols, num_cols = build_preprocessor(train_df)

    X_train = preprocessor.fit_transform(train_df.drop(columns=[TARGET]))
    X_test = preprocessor.transform(test_df.drop(columns=[TARGET]))
    print(f"  Encoded feature matrix: {X_train.shape}")

    # --- Logistic Regression baseline ---
    print("\nTraining Logistic Regression (class_weight='balanced' since "
          "positive class is only 11%)...")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_scores = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate("Logistic Regression", y_test, lr_scores)

    # --- XGBoost ---
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nTraining XGBoost (scale_pos_weight={pos_weight:.2f})...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        # min_child_weight requires more samples to justify a split --
        # directly guards against carving splits out of rare/noisy
        # categories (see rare-category consolidation in data_prep.py --
        # this is a second, complementary layer of defense against the
        # same overfitting failure mode).
        reg_lambda=2.0, min_child_weight=5,
    )
    xgb_model.fit(X_train, y_train)
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate("XGBoost", y_test, xgb_scores)

    print("\n=== Summary ===")
    print(pd.DataFrame({"LogReg": lr_metrics, "XGBoost": xgb_metrics}).round(3))

    import joblib
    joblib.dump(preprocessor, "../results/preprocessor.pkl")
    joblib.dump(xgb_model, "../results/xgb_model.pkl")
    np.save("../results/X_test_columns.npy",
            preprocessor.get_feature_names_out(), allow_pickle=True)
    print("\nSaved model + preprocessor to ../results/")


if __name__ == "__main__":
    main()
