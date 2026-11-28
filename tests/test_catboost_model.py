"""Synthetic smoke checks for the CatBoost model. Run with python -m unittest discover -s tests."""

import unittest

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.features.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features.split import get_cv
from src.models.catboost_model import (
    FEATURES, MISSING, CatBoostKSI, out_of_fold, to_catboost_frame,
)


class CatBoostModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(RANDOM_STATE)
        n_acc = 400
        groups = pd.Series(np.repeat(np.arange(n_acc), 2).astype(str))
        y = pd.Series(np.repeat(rng.integers(0, 2, size=n_acc), 2))
        X = pd.DataFrame({name: rng.choice(["1", "2", "3"], size=len(y)) for name in CATEGORICAL_FEATURES})
        for name in NUMERIC_FEATURES:
            X[name] = rng.normal(size=len(y))
        X["road_user"] = np.where(y == 1, "Pedestrian", "Car")
        X.loc[::7, "road_user"] = np.nan
        X.loc[::5, "age"] = np.nan
        X["extra_column"] = 1  # not a model feature: must be dropped
        cls.X, cls.y, cls.groups = X, y, groups

    def test_frame_has_m2_columns_and_no_missing_categories(self):
        frame = to_catboost_frame(self.X)
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertFalse(frame[CATEGORICAL_FEATURES].isna().any().any())
        self.assertTrue((frame.loc[::7, "road_user"] == MISSING).all())
        self.assertTrue(frame["age"].isna().any())  # numeric NaN left to CatBoost

    def test_out_of_fold_predicts_every_person_once(self):
        folds = list(get_cv().split(self.X, self.y, self.groups))
        oof, models = out_of_fold(self.X, self.y, folds, {"iterations": 30, "depth": 3}, keep_models=True)
        self.assertEqual(len(models), 5)
        self.assertFalse(np.isnan(oof).any())
        self.assertTrue(((oof >= 0) & (oof <= 1)).all())

    def test_sklearn_interface(self):
        model = CatBoostKSI(iterations=30, depth=3).fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.y), 2))
        self.assertEqual(set(model.predict(self.X)), {0, 1})
        self.assertEqual(model.get_params()["depth"], 3)


if __name__ == "__main__":
    unittest.main()
