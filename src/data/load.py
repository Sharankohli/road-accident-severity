# Owner: Sharan Raj JANAKIRAMAN (M1)
"""Load the raw ONISR CSV files exactly as they are, but in a consistent shape.

This step does NOT clean values (that is clean.py). It only fixes the
*file format* problems, so that every year looks the same:
  - separator is ';' and every value is quoted
  - 2022 'caract' file calls the accident key 'Accident_Id' instead of 'Num_Acc'
  - 2025 'vehicules' header has a line break inside 'id_vehicule'
  - many values have extra spaces, e.g. ' -1'
Everything is read as text (dtype=str), so codes like '01' are not silently
turned into numbers before we decide how to treat them.
"""
import pandas as pd

from src.config import RAW_DIR, YEARS

TABLES = ["caract", "lieux", "vehicules", "usagers"]


def read_table(table: str, year: int) -> pd.DataFrame:
    """Read one raw CSV (one table, one year) and return a DataFrame of strings."""
    if table not in TABLES:
        raise ValueError(f"Unknown table '{table}'. Choose from {TABLES}.")
    path = RAW_DIR / f"{table}-{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")

    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8", keep_default_na=False)

    # Clean the column names: remove line breaks and spaces, unify the key name.
    df.columns = [c.replace("\n", "").replace("\r", "").strip() for c in df.columns]
    df = df.rename(columns={"Accident_Id": "Num_Acc"})

    # Remove spaces around every value (' -1' -> '-1').
    df = df.apply(lambda col: col.str.strip())

    df["year"] = year
    return df


def load_table(table: str, years: list[int] = YEARS) -> pd.DataFrame:
    """Read one table for several years and stack them into one DataFrame."""
    return pd.concat([read_table(table, y) for y in years], ignore_index=True)
