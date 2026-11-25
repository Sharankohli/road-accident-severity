# Owner: Sharan Raj JANAKIRAMAN (M1)
"""Join the 4 cleaned tables into ONE table with one row per person.

Why one row per person: our question is "is THIS person killed or seriously
injured?", so the person is the unit of analysis. Each person gets the
circumstances of their accident (caract), its location (lieux) and the
vehicle they were in (vehicules).

How the tables link:
    usagers --Num_Acc-->     caract     (many people -> one accident)
    usagers --Num_Acc-->     lieux      (many people -> one location, see below)
    usagers --id_vehicule--> vehicules  (many people -> one vehicle)
Note: a pedestrian is attached to the vehicle that hit them, so for a
pedestrian the vehicle columns describe the vehicle that hit them.

The 'lieux' problem: since 2023 an accident can have several location rows
(e.g. an intersection of two roads: 47,346 accidents in 2023-2025). Joining
all of them would copy people 2 to 5 times. We keep ONE row per accident:
the road with the highest speed limit (vma), because speed drives crash
energy; ties keep the first row. Before 2023 there is only one row anyway.
"""
import pandas as pd

from src.config import YEARS
from src.data.clean import clean_table
from src.data.load import load_table


def main_location(lieux: pd.DataFrame) -> pd.DataFrame:
    """Keep one location row per accident: the one with the highest speed limit."""
    order = lieux.assign(_pos=range(len(lieux))).sort_values(
        ["Num_Acc", "vma", "_pos"], ascending=[True, False, True], na_position="last"
    )
    return order.drop_duplicates("Num_Acc", keep="first").drop(columns="_pos")


def build_persons(years: list[int] = YEARS) -> pd.DataFrame:
    """Return one row per person, with accident, location and vehicle columns.

    Output columns keep their original ONISR names. A new 'person_id' is added
    because the official 'id_usager' only exists from 2021.
    """
    caract = clean_table(load_table("caract", years))
    lieux = clean_table(load_table("lieux", years))
    vehicules = clean_table(load_table("vehicules", years))
    usagers = clean_table(load_table("usagers", years))

    persons = usagers.copy()
    persons.insert(0, "person_id", persons["year"].astype(str) + "_" + persons.groupby("year").cumcount().astype(str))

    # validate="many_to_one" makes pandas raise an error if the right table
    # has duplicate keys, so a silent row explosion is impossible.
    persons = persons.merge(caract.drop(columns="year"), on="Num_Acc", how="left", validate="many_to_one")
    persons = persons.merge(main_location(lieux).drop(columns="year"), on="Num_Acc", how="left", validate="many_to_one")
    persons = persons.merge(
        vehicules.drop(columns=["year", "Num_Acc", "num_veh"]), on="id_vehicule", how="left", validate="many_to_one"
    )

    if len(persons) != len(usagers):
        raise RuntimeError(f"Join changed the number of people: {len(usagers)} -> {len(persons)}")
    return persons
