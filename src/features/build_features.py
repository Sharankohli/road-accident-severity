"""Stateless, leakage-free person-level feature engineering."""

import numpy as np
import pandas as pd

from src.data.eda import road_user as _road_user


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow copy with row-wise features and no fitted state.

    Existing columns are shared, not modified; only derived columns are added.
    Call before ``get_splits``. Equipment flags use documented codes only:
    0 = none, 1 = belt, and 2 = helmet. Other codes remain unknown.
    """
    required = {"year", "an_nais", "hrmn", "mois", "jour", "dep", "catu", "catv", "secu1", "secu2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    out = df.copy(deep=False)
    year = pd.to_numeric(out["year"], errors="coerce")
    birth_year = pd.to_numeric(out["an_nais"], errors="coerce")
    out["age"] = year - birth_year

    hour = pd.to_numeric(out["hrmn"].astype("string").str.slice(0, 2), errors="coerce")
    hour = hour.where(hour.between(0, 23))
    out["hour"] = hour.astype("float64")
    out["night"] = np.where(hour.isna(), np.nan, ((hour >= 22) | (hour < 6)).astype(float))

    date = pd.to_datetime(
        {
            "year": year,
            "month": pd.to_numeric(out["mois"], errors="coerce"),
            "day": pd.to_numeric(out["jour"], errors="coerce"),
        },
        errors="coerce",
    )
    weekday = date.dt.dayofweek
    out["weekday"] = weekday.astype("Int64").astype("string")
    out["weekend"] = np.where(weekday.isna(), np.nan, weekday.ge(5).astype(float))

    users = _road_user(out)
    unknown_vehicle = out["catv"].isna() & out["catu"].ne("3")
    out["road_user"] = users.mask(unknown_vehicle, np.nan)

    equipment = out[["secu1", "secu2"]].astype("string")
    belt_or_helmet = equipment.isin({"1", "2"}).any(axis=1)
    explicitly_none = equipment["secu1"].eq("0") & (
        equipment["secu2"].isna() | equipment["secu2"].eq("0")
    )
    equipment_known = belt_or_helmet | explicitly_none
    out["belt_or_helmet_recorded"] = belt_or_helmet.astype("float64").where(equipment_known)
    out["equipment_none"] = explicitly_none.astype("float64").where(equipment_known)

    department = out["dep"].astype("string")
    out["is_overseas"] = department.str.startswith("97").astype("float64").where(department.notna())
    return out