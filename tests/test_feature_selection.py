"""Synthetic smoke checks. Run with python -m unittest discover -s tests."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, issparse
from sklearn.base import clone
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.features.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features.select import MutualInformationPreprocessor, build_selector
from src.features.split import get_cv
from src.models.baseline import logistic_baseline


class FeatureSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(RANDOM_STATE)
        groups = np.repeat(np.arange(500), 3)
        y = np.repeat(rng.integers(0, 2, size=500), 3)
        X = pd.DataFrame({
            name: rng.choice(list("abcdef"), size=len(y))
            for name in CATEGORICAL_FEATURES
        })
        for name in NUMERIC_FEATURES:
            X[name] = 0.0
        X["age"] = rng.normal(30 + 20 * y, 5)
        X.loc[::13, "age"] = np.nan
        X["night"] = (rng.random(len(y)) < 0.25 + 0.5 * y).astype(float)
        X["road_user"] = np.where(y, "pedestrian", "car")
        folds = list(get_cv().split(X, y, groups))
        # This category and missingness indicator are absent when fold 0 validates.
        X.loc[folds[0][1], "dep"] = "fold-zero-only"
        X.loc[folds[0][1], "nbv"] = np.nan
        cls.X, cls.y, cls.groups, cls.folds = X, y, groups, folds

    def make_pipeline(self, method):
        baseline = logistic_baseline()
        prep = baseline.named_steps["prep"]
        if method == "mi":
            prep = MutualInformationPreprocessor([("columns", prep)])
        selector = build_selector(method, **({"k": 12} if method == "mi" else {}))
        return Pipeline([("prep", prep), ("select", selector), ("model", baseline.named_steps["model"])])

    def test_grouped_oof_fits_and_feature_names(self):
        X, y, groups, folds = self.X, self.y, self.groups, self.folds
        for train, valid in folds:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
        for method in ("mi", "l1"):
            with self.subTest(method=method):
                pipeline = self.make_pipeline(method)
                selector_type = type(pipeline.named_steps["select"])
                original_fit = selector_type.fit
                fits = []
                dense_widths = []
                original_toarray = csc_matrix.toarray

                def record_fit(selector, matrix, target, **kwargs):
                    train, _ = folds[len(fits)]
                    self.assertTrue(issparse(matrix))
                    np.testing.assert_array_equal(target, y[train])
                    expected = clone(pipeline.named_steps["prep"]).fit_transform(X.iloc[train], y[train])
                    self.assertEqual((matrix != expected).nnz, 0)
                    result = original_fit(selector, matrix, target, **kwargs)
                    fits.append(selector.get_support())
                    return result

                def record_toarray(matrix, *args, **kwargs):
                    dense_widths.append(matrix.shape[1])
                    return original_toarray(matrix, *args, **kwargs)

                with patch.object(selector_type, "fit", record_fit), patch.object(
                    csc_matrix, "toarray", record_toarray
                ), patch.object(csr_matrix, "toarray", record_toarray):
                    probabilities = cross_val_predict(
                        pipeline, X, y, cv=folds, method="predict_proba", n_jobs=1
                    )
                self.assertEqual(probabilities.shape, (len(y), 2))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertEqual(len(fits), 5)
                self.assertGreater(len({len(mask) for mask in fits}), 1)
                self.assertTrue(all(0 < mask.sum() < len(mask) for mask in fits))
                if method == "mi":
                    self.assertTrue(dense_widths)
                    self.assertEqual(set(dense_widths), {1})
                    self.assertTrue(all(mask.sum() == 12 for mask in fits))

                repeated = cross_val_predict(
                    clone(pipeline), X, y, cv=folds, method="predict_proba", n_jobs=1
                )
                np.testing.assert_allclose(probabilities, repeated)
                train, _ = folds[1]
                fitted = clone(pipeline).fit(X.iloc[train], y[train])
                prep, selector = fitted.named_steps["prep"], fitted.named_steps["select"]
                names = prep.get_feature_names_out()
                np.testing.assert_array_equal(fitted[:-1].get_feature_names_out(), names[selector.get_support()])
                self.assertTrue(issparse(fitted[:-1].transform(X.iloc[train])))
                if method == "mi":
                    names = names.tolist()
                    for name in ("night", "missingindicator_age", "road_user_car"):
                        self.assertTrue(selector.discrete_mask_[names.index(name)])
                    for name in ("age", "nbv"):
                        self.assertFalse(selector.discrete_mask_[names.index(name)])
                    self.assertEqual(selector.scores_[names.index("hour")], 0)
                    self.assertEqual(selector.scores_[names.index("weekend")], 0)
                    self.assertAlmostEqual(
                        selector.scores_[names.index("night")],
                        mutual_info_score(X.iloc[train]["night"].astype(int), y[train]),
                    )

    def test_invalid_settings_and_empty_l1(self):
        for kwargs in (
            {"method": "unknown"}, {"method": "mi"}, {"method": "mi", "k": True},
            {"method": "mi", "k": 0}, {"method": "mi", "k": 1.5},
            {"k": 4}, {"C": 0}, {"C": -1}, {"C": np.inf}, {"C": np.nan}, {"C": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_selector(**kwargs)
        matrix = logistic_baseline().named_steps["prep"].fit_transform(self.X)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_selector("mi", k=matrix.shape[1] + 1).fit(matrix, self.y)
        with self.assertRaisesRegex(ValueError, "fold-specific mask"):
            build_selector("mi", k=1).fit(matrix, self.y)
        with self.assertRaisesRegex(ValueError, "zero features"):
            build_selector(C=1e-9).fit(matrix, self.y)


if __name__ == "__main__":
    unittest.main()
