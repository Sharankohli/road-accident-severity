"""Shared year-based train/test split and accident-grouped cross-validation."""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import GROUP_COL, RANDOM_STATE, TARGET, TEST_YEARS, TRAIN_YEARS

_NEVER_FEATURES = {
    TARGET,
    "grav",
    "person_id",
    GROUP_COL,
    "id_vehicule",
    "id_usager",
    "num_veh",
    "year",
    "an",
    "jour",
    "hrmn",
}


def get_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Return X_train, y_train, groups_train, X_test, y_test for the agreed years.

    Call ``add_features`` before this function when derived features are needed.
    The returned X frames never contain the target, outcome code, identifiers,
    or raw year/day/time fields. Accident IDs are returned separately for CV.
    """
    required = {"year", TARGET, GROUP_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if df[["year", TARGET, GROUP_COL]].isna().any().any():
        raise ValueError("year, target, and accident group must not contain missing values")

    years = pd.to_numeric(df["year"], errors="coerce")
    if years.isna().any():
        raise ValueError("year must contain valid numeric years")
    allowed_years = set(TRAIN_YEARS).union(TEST_YEARS)
    unexpected_years = set(years.unique()).difference(allowed_years)
    if unexpected_years:
        raise ValueError(f"Unsupported years in input: {sorted(unexpected_years)}")

    train_mask = years.isin(TRAIN_YEARS)
    test_mask = years.isin(TEST_YEARS)
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Input must contain at least one training row and one test row")

    train_groups = df.loc[train_mask, GROUP_COL]
    test_groups = df.loc[test_mask, GROUP_COL]
    overlap = pd.Index(train_groups.unique()).intersection(test_groups.unique())
    if not overlap.empty:
        raise ValueError(f"Accident groups cross the year split ({len(overlap)} overlaps)")

    forbidden = [column for column in _NEVER_FEATURES if column in df.columns]
    feature_columns = [column for column in df.columns if column not in forbidden]
    X_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, TARGET].copy()
    groups_train = train_groups.copy()
    X_test = df.loc[test_mask, feature_columns]
    y_test = df.loc[test_mask, TARGET].copy()

    target_values = set(pd.concat([y_train, y_test]).unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1; found {sorted(target_values)}")
    return X_train, y_train, groups_train, X_test, y_test


def get_cv() -> StratifiedGroupKFold:
    """Return the shared shuffled, stratified 5-fold accident-grouped splitter."""
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)