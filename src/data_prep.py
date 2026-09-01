"""
Cleans the raw UCI Diabetes 130-US Hospitals dataset and produces a
patient-level train/test split for 30-day readmission prediction.

Key decisions, explained inline:
  - Target: readmitted == '<30' (the CMS-penalized outcome) vs everything
    else. This throws away the >30-vs-NO distinction, which is a real
    simplification -- defensible because <30 is the specific outcome
    hospitals are financially and clinically accountable for.
  - Split by patient_nbr, not by row, to prevent leakage (see EDA above).
  - diag_1/2/3 are raw ICD-9 codes (hundreds of distinct values) -- we
    group them into clinically meaningful categories, which is standard
    practice in the literature on this dataset (Strack et al. 2014, the
    paper this dataset is from, does the same).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW_PATH = "../data/diabetic_data.csv"


def _map_icd9_to_category(code: str) -> str:
    """
    ICD-9 codes are mostly numeric with clinically meaningful ranges
    (e.g. 390-459 = circulatory system). A few are V/E codes (external
    causes / supplemental) which we bucket as 'Other'. This mirrors the
    9-category grouping used in the original paper for this dataset.
    """
    if pd.isna(code) or code == "?":
        return "Missing"
    if code.startswith(("V", "E")):
        return "Other"
    try:
        val = float(code)
    except ValueError:
        return "Other"
    if 390 <= val <= 459 or val == 785:
        return "Circulatory"
    if 460 <= val <= 519 or val == 786:
        return "Respiratory"
    if 520 <= val <= 579 or val == 787:
        return "Digestive"
    if int(val) == 250 or (250 <= val < 251):
        return "Diabetes"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


AGE_ORDER = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
             "[60-70)", "[70-80)", "[80-90)", "[90-100)"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = df.replace("?", np.nan)

    # Binary target: the CMS-penalized outcome.
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)

    # Drop columns that are unusable (97% missing) or not real features.
    df = df.drop(columns=["weight", "readmitted", "encounter_id"])

    # ICD-9 diagnosis grouping -- goes from ~700 distinct codes/column to
    # 9 clinically meaningful buckets, which both reduces overfitting risk
    # and makes SHAP explanations readable ("Circulatory" means something
    # to a clinician; ICD9 code "428.0" less so at a glance).
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(_map_icd9_to_category)

    # Age is given as 10-year bins -- encode as ordinal (a clinician's
    # intuition that risk trends with age is a real, directional prior;
    # one-hot would throw that ordering away).
    df["age_ordinal"] = df["age"].apply(lambda a: AGE_ORDER.index(a))

    # medical_specialty / payer_code / race: keep missingness as its own
    # category rather than imputing -- missingness itself can be
    # informative (e.g. certain payer types correlate with which data
    # gets recorded) and imputing would fabricate information we don't have.
    for col in ["medical_specialty", "payer_code", "race"]:
        df[col] = df[col].fillna("Missing")

    return df


def patient_level_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """GroupShuffleSplit keyed on patient_nbr -- guarantees no patient's
    encounters appear in both train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_nbr"]))
    train_df = df.iloc[train_idx].drop(columns=["patient_nbr"])
    test_df = df.iloc[test_idx].drop(columns=["patient_nbr"])
    return train_df, test_df


RARE_CATEGORY_MIN_COUNT = 500  # ~0.6% of the 81k training rows


def consolidate_rare_categories(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                 min_count: int = RARE_CATEGORY_MIN_COUNT):
    """
    Buckets any categorical level seen fewer than `min_count` times in
    TRAIN into 'Other'. Fit strictly on train (never test) to avoid
    leakage, then apply the identical mapping to test.

    Why this matters: a categorical level with e.g. 50 examples gives a
    tree model just enough rows to carve out a split that fits training
    noise, not signal (see the SHAP diagnosis in chat -- 'Otolaryngology'
    with 99 patients was outranking number_inpatient, a well-established
    real predictor, in global feature importance -- a classic overfitting
    tell, not a clinical finding).
    """
    cat_cols = [c for c in train_df.columns
                if not pd.api.types.is_numeric_dtype(train_df[c]) and c != "readmit_30"]
    for col in cat_cols:
        counts = train_df[col].value_counts()
        keep = set(counts[counts >= min_count].index)
        train_df[col] = train_df[col].where(train_df[col].isin(keep), "Other")
        test_df[col] = test_df[col].where(test_df[col].isin(keep), "Other")
    return train_df, test_df


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Cleaned shape: {df.shape}")
    print(f"Readmit-within-30-days rate: {df['readmit_30'].mean():.3%}")

    train_df, test_df = patient_level_split(df)
    train_df, test_df = consolidate_rare_categories(train_df, test_df)
    print(f"\nTrain: {len(train_df)} encounters, "
          f"readmit rate={train_df['readmit_30'].mean():.3%}")
    print(f"Test:  {len(test_df)} encounters, "
          f"readmit rate={test_df['readmit_30'].mean():.3%}")

    # Sanity check: verify zero patient overlap between splits.
    df_full = load_and_clean()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_full, groups=df_full["patient_nbr"]))
    train_patients = set(df_full.iloc[train_idx]["patient_nbr"])
    test_patients = set(df_full.iloc[test_idx]["patient_nbr"])
    overlap = train_patients & test_patients
    print(f"\nPatient overlap between train/test: {len(overlap)} (must be 0)")

    train_df.to_csv("../data/train.csv", index=False)
    test_df.to_csv("../data/test.csv", index=False)
    print("\nSaved train.csv / test.csv")
