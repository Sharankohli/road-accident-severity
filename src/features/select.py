"""Fold-fitted feature selection for the Week 2 preprocessing pipeline."""

from numbers import Integral, Real

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE

# Binary engineered fields and missingness indicators are discrete even after scaling.
_CONTINUOUS_FEATURES = {"age", "hour", "nbv", "vma"}


class MutualInformationPreprocessor(Pipeline):
    """Wrap the baseline preprocessor to carry fold-specific types into MI fitting.

    Use as `MutualInformationPreprocessor([("columns", baseline_prep)])`.
    Only the training matrix needs the mask; inference uses the usual transform.
    """

    def fit_transform(self, X, y=None, **params):
        matrix = super().fit_transform(X, y, **params)
        if not issparse(matrix) or matrix.format != "csr":
            matrix = csr_matrix(matrix)
        matrix.mi_discrete_mask = ~np.isin(
            self.get_feature_names_out(), sorted(_CONTINUOUS_FEATURES)
        )
        return matrix


def _mixed_mutual_information(X, y, discrete):
    columns = X.tocsc()
    scores = np.zeros(X.shape[1])
    if discrete.any():
        binary = columns[:, discrete]
        # The baseline centers binary numerics: the upper level is positive.
        # One-hot values are already 0/1. Use integer labels for sklearn's MI.
        binary.data = (binary.data > 0).astype(np.int8)
        binary.eliminate_zeros()
        scores[discrete] = mutual_info_classif(
            binary, y, discrete_features=True, random_state=RANDOM_STATE, n_jobs=1
        )
    for index in np.flatnonzero(~discrete):
        values = columns[:, index:index + 1].toarray()
        if np.ptp(values) > 0:
            scores[index] = mutual_info_classif(
                values, y, discrete_features=False, random_state=RANDOM_STATE, n_jobs=1
            )[0]
    return scores


class _MutualInformationSelector(SelectKBest):
    def __init__(self, k):
        super().__init__(self._score, k=k)

    def fit(self, X, y=None):
        if isinstance(self.k, bool) or not isinstance(self.k, Integral) or self.k < 1:
            raise ValueError("MI requires a positive integer k.")
        if self.k > X.shape[1]:
            raise ValueError(f"k={self.k} exceeds the {X.shape[1]} features in this fold.")
        discrete = getattr(X, "mi_discrete_mask", None)
        if discrete is None or discrete.shape != (X.shape[1],):
            raise ValueError("MI requires the fold-specific mask from MutualInformationPreprocessor.")
        self.discrete_mask_ = discrete
        return super().fit(X, y)

    def _score(self, X, y):
        return _mixed_mutual_information(X, y, self.discrete_mask_)


class _NonemptySelectFromModel(SelectFromModel):
    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        if not self.get_support().any():
            raise ValueError("L1 selected zero features. Increase C and retry.")
        return self


def build_selector(method="l1", k=None, C=0.05):
    """Return an unfitted feature selector for use inside a Pipeline.

    MI requires a positive integer k and a MutualInformationPreprocessor.
    L1 uses C as inverse regularization strength and derives the feature count.
    """
    if method == "mi":
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("MI requires a positive integer k.")
        return _MutualInformationSelector(k)
    if method != "l1":
        raise ValueError("method must be 'mi' or 'l1'.")
    if k is not None:
        raise ValueError("k applies only to MI.")
    if isinstance(C, bool) or not isinstance(C, Real) or not np.isfinite(C) or C <= 0:
        raise ValueError("L1 requires a positive finite C.")
    estimator = LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=C,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return _NonemptySelectFromModel(estimator, threshold=1e-5)
