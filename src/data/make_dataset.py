# Owner: Sharan Raj JANAKIRAMAN (M1)
"""Build the final M1 output files that the rest of the team uses.

Run from the repo root:   python -m src.data.make_dataset

Outputs (in data/processed/, not on GitHub):
  persons.parquet         one row per person, all years, target KSI, no grav
  persons_sample.parquet  a small random sample of whole accidents (~2,000 people)
                          so teammates can start coding before the full file is ready
"""
import numpy as np
import pandas as pd

from src.config import GROUP_COL, PROCESSED_DIR, RANDOM_STATE, TARGET
from src.data.join import build_persons

# Columns removed, with the reason. Keeping reasons in code makes the choice defendable.
DROP = {
    "grav": "the outcome itself -> replaced by KSI (keeping it = target leakage)",
    "id_usager": "official person ID, only from 2021 -> replaced by person_id",
    "num_veh": "vehicle letter inside the accident (A, B, ...), no meaning",
    "an": "same as year",
    "adr": "free-text address",
    "voie": "free-text road number/name",
    "v1": "road number suffix (bis, ter)",
    "v2": "road letter suffix",
    "pr": "road marker number (location detail, no meaning across roads)",
    "pr1": "distance to road marker",
}

SAMPLE_PEOPLE = 2_000


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add KSI (1 = killed or hospitalised > 24 h, else 0) and remove people with unknown severity.

    grav codes: 1 unharmed, 2 killed, 3 hospitalised, 4 slightly injured.
    The codes are NOT in severity order, so we map them explicitly.
    """
    known = df[df["grav"].notna()].copy()
    known[TARGET] = known["grav"].isin(["2", "3"]).astype(np.int8)
    return known


def sample_accidents(df: pd.DataFrame, n_people: int = SAMPLE_PEOPLE) -> pd.DataFrame:
    """Random sample of WHOLE accidents (people of one accident stay together)."""
    acc = df[GROUP_COL].drop_duplicates().sample(frac=1, random_state=RANDOM_STATE)
    sizes = df.groupby(GROUP_COL).size().loc[acc]
    keep = sizes.index[sizes.cumsum() <= n_people]
    return df[df[GROUP_COL].isin(keep)]


def main() -> None:
    persons = make_target(build_persons())
    persons = persons.drop(columns=list(DROP))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    persons.to_parquet(PROCESSED_DIR / "persons.parquet", index=False)
    sample = sample_accidents(persons)
    sample.to_parquet(PROCESSED_DIR / "persons_sample.parquet", index=False)

    print(f"persons.parquet        {len(persons):,} people, {persons.shape[1]} columns, "
          f"KSI rate {persons[TARGET].mean():.1%}")
    print(f"persons_sample.parquet {len(sample):,} people from {sample[GROUP_COL].nunique():,} accidents")


if __name__ == "__main__":
    main()
