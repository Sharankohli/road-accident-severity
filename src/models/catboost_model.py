"""Native categorical CatBoost pipeline; all fitted state stays inside folds."""
import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.config import RANDOM_STATE
from src.features.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES


class CatBoostFrame(TransformerMixin, BaseEstimator):
    """Stateless schema conversion, with native numeric NaNs and string categories."""
    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        return self

    def transform(self, X):
        frame = X[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        for col in NUMERIC_FEATURES:
            frame[col] = frame[col].astype(float)
        for col in CATEGORICAL_FEATURES:
            frame[col] = frame[col].astype('string').fillna('__MISSING__').astype(str)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.array(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def catboost_pipeline(**params):
    settings = dict(iterations=500, depth=6, learning_rate=0.1, l2_leaf_reg=3,
                    loss_function='Logloss', eval_metric='PRAUC',
                    auto_class_weights='Balanced', random_seed=RANDOM_STATE,
                    thread_count=4, verbose=False, allow_writing_files=False,
                    # Explicit: the paper's ordered boosting, not CPU's large-data Plain default.
                    boosting_type='Ordered', cat_features=tuple(CATEGORICAL_FEATURES))
    settings.update(params)
    return Pipeline([('prep', CatBoostFrame()), ('model', CatBoostClassifier(**settings))])


def catboost_candidates():
    # A small, predeclared joint search; it is not an exhaustive grid.
    return {
        'depth4_fast': catboost_pipeline(depth=4, learning_rate=0.1, iterations=500, l2_leaf_reg=3),
        'depth6_slow': catboost_pipeline(depth=6, learning_rate=0.03, iterations=750, l2_leaf_reg=5),
        'depth8_fast': catboost_pipeline(depth=8, learning_rate=0.1, iterations=500, l2_leaf_reg=5),
        'depth6_unweighted': catboost_pipeline(depth=6, learning_rate=0.1, iterations=500, auto_class_weights=None),
    }
