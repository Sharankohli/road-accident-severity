# Owner: Sharan Raj JANAKIRAMAN (M1)
"""Small helpers for the EDA notebook (01_data_audit_eda.ipynb).

These group raw codes into readable categories ONLY to describe the data.
They are not model features: feature engineering belongs to M2.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson confidence interval for a proportion k/n.

    Better than the simple p +/- 1.96*sqrt(p(1-p)/n) when p is small or n is small,
    because it never goes below 0 or above 1.
    """
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def rate_table(df: pd.DataFrame, col: str, target: str = "KSI", min_n: int = 300) -> pd.DataFrame:
    """KSI rate per category of `col`, with count and 95% CI, sorted by rate."""
    g = df.groupby(col, observed=True)[target].agg(n="size", k="sum")
    g = g[g.n >= min_n]
    g["rate"] = g.k / g.n
    g["ci_low"], g["ci_high"] = wilson_ci(g.k.to_numpy(), g.n.to_numpy())
    return g.sort_values("rate", ascending=False)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Strength of association between two categorical variables, from 0 (none) to 1 (perfect).

    Based on the chi-square statistic: V = sqrt(chi2 / (n * (min(rows, cols) - 1))).
    Missing values are kept as their own category so they are not silently dropped.
    """
    table = pd.crosstab(x.astype("string").fillna("missing"), y.astype("string").fillna("missing"))
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, c = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0


def road_user(df: pd.DataFrame) -> pd.Series:
    """Readable road-user type: pedestrian, or the vehicle the person was in."""
    catv = df["catv"].astype("string")
    groups = {
        "Bicycle": {"1"}, "E-bike": {"80"}, "E-scooter / personal device": {"50", "60"},
        "Moped <50cc": {"2", "30", "41"}, "Motorcycle >50cc": {"31", "32", "33", "34", "42", "43"},
        "Car": {"7"}, "Van": {"10"}, "Heavy goods vehicle": {"13", "14", "15", "16", "17"},
        "Bus / coach": {"37", "38"},
    }
    out = pd.Series("Other / unknown", index=df.index)
    for name, codes in groups.items():
        out[catv.isin(codes)] = name
    out[df["catu"] == "3"] = "Pedestrian"
    return out


def age_band(df: pd.DataFrame) -> pd.Series:
    age = df["year"] - df["an_nais"]
    return pd.cut(age, [-1, 17, 24, 44, 64, 74, 120], labels=["0-17", "18-24", "25-44", "45-64", "65-74", "75+"])
