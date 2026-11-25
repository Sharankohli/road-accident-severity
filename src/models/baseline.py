# Owner: Sharan Raj JANAKIRAMAN (M3)
"""Baseline models: the minimum scores every later model must beat.

1. Majority class: always predicts the KSI rate learned on training data
   (scikit-learn DummyClassifier). It never says "KSI".
2. Expert rule: KSI if pedestrian, OR motorcyclist (>50cc), OR speed limit >= 80 km/h.
   Written by hand from road-safety knowledge, not learned: nothing to fit.
3. Logistic regression: added when the M2 preprocessor is ready.

Both classes follow the scikit-learn interface (fit / predict / predict_proba),
so they go through exactly the same evaluation code as real models.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

MOTORCYCLE_CODES = {"31", "32", "33", "34", "42", "43"}  # catv codes for motorised 2/3-wheelers > 50cc


def majority_baseline() -> DummyClassifier:
    """Predicts the training KSI rate for everyone (strategy='prior')."""
    return DummyClassifier(strategy="prior")


class ExpertRuleBaseline(ClassifierMixin, BaseEstimator):
    """KSI = 1 if pedestrian, motorcyclist (>50cc) or speed limit >= 80 km/h.

    Uses raw columns as they are in persons.parquet (catu, catv, vma),
    so it needs no preprocessing. Missing values count as "condition not met".
    """

    def __init__(self, min_speed: float = 80):
        self.min_speed = min_speed

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])  # nothing is learned
        return self

    def predict(self, X):
        pedestrian = X["catu"].eq("3")
        motorcycle = X["catv"].isin(MOTORCYCLE_CODES)
        fast_road = X["vma"].ge(self.min_speed)
        return (pedestrian | motorcycle | fast_road).astype(int).to_numpy()

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])
