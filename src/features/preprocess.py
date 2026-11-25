"""Fold-fitted preprocessing for linear and tree-based estimators."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "hour",
    "weekend",
    "night",
    "belt_or_helmet_recorded",
    "equipment_none",
    "is_overseas",
    "nbv",
    "vma",
]

CATEGORICAL_FEATURES = [
    "place",
    "catu",
    "sexe",
    "trajet",
    "secu1",
    "secu2",
    "locp",
    "actp",
    "etatp",
    "mois",
    "weekday",
    "lum",
    "dep",
    "agg",
    "int",
    "atm",
    "col",
    "catr",
    "circ",
    "vosp",
    "prof",
    "plan",
    "surf",
    "infra",
    "situ",
    "senc",
    "obs",
    "obsm",
    "choc",
    "manv",
    "motor",
    "road_user",
]


def build_preprocessor(kind: str = "linear") -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for ``linear`` or ``tree`` models.

    Only the declared columns are passed through; IDs, outcomes, raw dates,
    coordinates, ``com``, ``catv``, and drift-prone ``secu3`` are excluded.
    Fit this object only as a step inside a model Pipeline.
    """
    if kind not in {"linear", "tree"}:
        raise ValueError("kind must be either 'linear' or 'tree'")

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        ),
    ]
    if kind == "linear":
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)

    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )