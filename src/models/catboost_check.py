# Owner: FOTSING TANKOU DAN KATEL (M4b)
"""CatBoost: the out-of-course algorithm of the project.

Paper: L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, A. Gulin.
"CatBoost: unbiased boosting with categorical features." NeurIPS 2018.
https://arxiv.org/abs/1706.09516

Why CatBoost here: most of our columns are categories (vehicle, road, weather,
department ...). CatBoost turns each category into a number with "ordered
target statistics": a row only uses the target of the rows placed before it in
a random order, so its own label never leaks into its encoding. No one-hot
encoding is needed. Note: with catboost 1.2.10 on CPU the default boosting_type
is "Plain", so the paper's second idea (ordered boosting) is not used here;
ordered target statistics for the categories are always used.

Fair comparison with the other models:
  - same columns as M2's preprocessor (NUMERIC_FEATURES + CATEGORICAL_FEATURES),
  - same accident-grouped folds, PR-AUC as main score, evaluate() for the rest,
  - one CatBoost model fitted per fold (CatBoost is not a scikit-learn Pipeline,
    but the only preparation, `to_catboost_frame`, is row-wise and learns nothing).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score

from src.config import RANDOM_STATE
from src.features.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MISSING = "missing"


def to_catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Keep the M2 columns; categories as strings with NaN -> "missing".

    Numeric NaN are left as NaN: CatBoost handles them itself (treated as
    smaller than every value). Row-wise only, so it is safe before CV.
    """
    out = X[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("string").fillna(MISSING).astype(str)
    return out


class CatBoostKSI(ClassifierMixin, BaseEstimator):
    """CatBoostClassifier with the column preparation built in.

    Follows the scikit-learn interface (fit / predict_proba / get_params), so it
    can be used with permutation_importance and the shared evaluation code.
    """

    def __init__(self, depth=6, learning_rate=0.1, iterations=1000, l2_leaf_reg=3.0,
                 auto_class_weights="Balanced", thread_count=-1):
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.thread_count = thread_count

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            l2_leaf_reg=self.l2_leaf_reg,
            auto_class_weights=self.auto_class_weights,
            cat_features=CATEGORICAL_FEATURES,
            eval_metric="PRAUC",
            random_seed=RANDOM_STATE,
            thread_count=self.thread_count,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(to_catboost_frame(X), np.asarray(y))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X, ntree_end=0):
        """ntree_end > 0 uses only the first trees (score a smaller iterations value)."""
        return self.model_.predict_proba(to_catboost_frame(X), ntree_end=ntree_end)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def tune(X, y, folds, grid, check_iterations=(500, 1000), progress_file=None):
    """Grouped CV over `grid` (list of dicts). Returns one row per setting.

    Each model is trained with max(check_iterations) trees; the smaller values
    are scored with the first trees of the same model (ntree_end). No early
    stopping on the validation fold, so the validation score stays honest.
    If `progress_file` is given, each finished setting is saved there and
    skipped on the next run (a long run can restart after a crash).
    """
    done = pd.read_csv(progress_file) if progress_file and Path(progress_file).exists() else pd.DataFrame()
    rows = []
    n_max = max(check_iterations)
    for params in grid:
        key = json.dumps(params, sort_keys=True)
        if not done.empty and set(done.loc[done.setting == key, "iterations"]) >= set(check_iterations):
            old = done[(done.setting == key) & done.iterations.isin(check_iterations)]
            rows += [{**params, **r} for r in old[["iterations", "pr_auc_mean", "pr_auc_std"]].to_dict("records")]
            print(f"  {params} -> already done")
            continue
        scores = {n: [] for n in check_iterations}
        for train, valid in folds:
            model = CatBoostKSI(iterations=n_max, **params).fit(X.iloc[train], y.iloc[train])
            for n in check_iterations:
                proba = model.predict_proba(X.iloc[valid], ntree_end=n)[:, 1]
                scores[n].append(average_precision_score(y.iloc[valid], proba))
        new = [{**params, "iterations": n, "pr_auc_mean": np.mean(scores[n]), "pr_auc_std": np.std(scores[n])}
               for n in check_iterations]
        rows += new
        if progress_file:
            done = pd.concat([done, pd.DataFrame(new)[["iterations", "pr_auc_mean", "pr_auc_std"]].assign(setting=key)])
            done.to_csv(progress_file, index=False)
        print(f"  {params} -> " + ", ".join(f"{n}: {np.mean(s):.4f}" for n, s in scores.items()))
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def out_of_fold(X, y, folds, params, keep_models=False):
    """Out-of-fold KSI probabilities: each person predicted by a model that never saw their accident."""
    oof = np.full(len(y), np.nan)
    models = []
    for k, (train, valid) in enumerate(folds, start=1):
        model = CatBoostKSI(**params).fit(X.iloc[train], y.iloc[train])
        oof[valid] = model.predict_proba(X.iloc[valid])[:, 1]
        print(f"  fold {k}: PR-AUC {average_precision_score(y.iloc[valid], oof[valid]):.4f}")
        if keep_models:
            models.append(model)
    return oof, models
