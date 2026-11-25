# Owner: Sharan Raj JANAKIRAMAN (M1)
"""Turn the many raw "missing" codes into one standard missing value (NaN).

Why this matters: a model reads "-1" as a real number or a real category.
If -1 means "the police did not fill this in", the model would learn from
noise. So every code that means "unknown" becomes NaN.

The rules come from the official ONISR data dictionary
("Description des bases de données annuelles", 2026), checked column by column:
  - '-1', '', '.' and 'N/A' always mean "not filled in"  -> NaN
  - '0' is TRICKY: in some columns it is a real answer, in others it means unknown.
      real answer : obs 0 = "no fixed obstacle", secu1 0 = "no equipment", ...
      unknown     : catv 0 = "undeterminable", manv 0 = "unknown", ...
    Turning every 0 into NaN would delete real information (e.g. 557,062
    vehicles that hit no fixed obstacle), so 0 is handled per column.
"""
import numpy as np
import pandas as pd

ALWAYS_MISSING = {"", ".", "-1", "N/A"}

# Columns where 0 is a real category -> keep "0".
ZERO_IS_REAL = {
    "vosp",    # 0 = no reserved lane
    "infra",   # 0 = no special infrastructure
    "situ",    # 0 = none
    "obs",     # 0 = no fixed obstacle hit
    "obsm",    # 0 = no moving obstacle hit
    "choc",    # 0 = no impact point
    "secu1", "secu2", "secu3",  # 0 = no safety equipment
    "locp",    # 0 = not applicable (person is not a pedestrian)
}

# Columns where 0 means unknown / not filled in -> NaN.
# (For coded columns with no 0 category, the dictionary says 0 = not filled in.)
ZERO_IS_MISSING = {
    "senc", "catv", "manv", "motor", "trajet", "actp",  # 0 = unknown / not filled
    "nbv", "vma",                                        # 0 lanes or 0 km/h is impossible
    "lum", "agg", "int", "atm", "col", "catr", "circ", "prof", "plan", "surf",
    "place", "catu", "grav", "sexe", "etatp",
}

# Columns that are numbers, not category codes.
NUMERIC = {"lat", "long", "nbv", "vma", "larrout", "lartpc", "an_nais", "occutc"}

# Identifiers and free text: never touched.
UNTOUCHED = {"Num_Acc", "id_vehicule", "id_usager", "num_veh", "year",
             "adr", "voie", "v1", "v2", "pr", "pr1", "com", "dep", "hrmn",
             "jour", "mois", "an"}


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a raw table with all missing codes replaced by NaN.

    Input : a DataFrame from load.load_table() (all values are text).
    Output: same columns; coded columns stay text (e.g. '7'), numeric columns
            become floats, and every "unknown" code becomes NaN.
    """
    out = df.copy()
    for col in out.columns:
        if col in UNTOUCHED or col.lower() in UNTOUCHED:
            continue
        s = out[col].astype("string")

        # Codes are written without leading zeros in the data ('7', not '07').
        # Normalise anyway, so '07' and '7' can never become two categories.
        if col not in NUMERIC:
            s = s.str.lstrip("0").replace("", "0").where(s != "", "")

        missing = s.isin(ALWAYS_MISSING)
        if col in ZERO_IS_MISSING:
            missing |= s == "0"
        s = s.mask(missing)

        if col in NUMERIC:
            # French decimals use a comma: '49,5185' -> 49.5185
            num = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
            out[col] = num.astype("float64")  # plain float: works with every library
        else:
            out[col] = s.astype(object).where(s.notna(), np.nan)
    return out
