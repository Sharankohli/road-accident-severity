"""Statistical and leakage checks for M4, using small synthetic inputs."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupKFold
from src.evaluation.bootstrap import _WeightedMetrics, bootstrap_ci, bootstrap_comparison
from src.evaluation.metrics import evaluate
from src.models.m4_common import accident_indices, validate_folds
from src.models.catboost_model import CatBoostFrame
from src.features.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_weighted_metrics_equal_literal_duplicate_accidents_with_ties():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    p = np.array([.1, .5, .5, .7, .9, .2, .7, .1])
    groups = np.repeat(np.arange(4), 2)
    rng = np.random.default_rng(42)
    for threshold in [0, .5, .7, 1]:
        scorer = _WeightedMetrics(y, p, threshold)
        for _ in range(20):
            drawn = rng.integers(0, 4, size=4)
            ix = np.concatenate([np.flatnonzero(groups == g) for g in drawn])
            if len(np.unique(y[ix])) < 2:
                continue
            weights = np.bincount(drawn, minlength=4)[groups]
            actual, expected = scorer(weights), evaluate(y[ix], p[ix], threshold)
            for metric in expected:
                assert actual[metric] == pytest.approx(expected[metric], abs=1e-12)


def test_bootstrap_reproducibility_paired_and_input_validation():
    y = np.tile([0, 1], 10)
    p = np.linspace(.05, .95, 20)
    groups = np.repeat(np.arange(10), 2)
    a = bootstrap_ci(y, p, groups, n_boot=25)
    pd.testing.assert_frame_equal(a, bootstrap_ci(y, p, groups, n_boot=25))
    ci, paired = bootstrap_comparison(y, {'a': p, 'b': p}, groups, n_boot=25, reference='a')
    assert np.allclose(paired[['estimate', 'low', 'high']], 0)
    assert (ci.low <= ci.high).all()
    for args in [(y, p[:-1], groups), (np.zeros(20), p, groups), (y, p + 1, groups)]:
        with pytest.raises(ValueError):
            bootstrap_ci(*args, n_boot=10)
    with pytest.raises(ValueError, match='indexes'):
        bootstrap_ci(pd.Series(y), pd.Series(p, index=np.arange(20)[::-1]), pd.Series(groups), n_boot=10)


def test_accident_subsample_and_fold_validation():
    groups = pd.Series(np.repeat(np.arange(30), np.arange(30) % 4 + 1))
    ix = accident_indices(groups, max_people=25)
    assert len(ix) <= 25
    for g in groups.iloc[ix].unique():
        assert (groups.iloc[ix] == g).sum() == (groups == g).sum()
    folds = list(GroupKFold(5).split(np.zeros(len(groups)), groups=groups))
    validate_folds(folds, groups)
    with pytest.raises(ValueError):
        validate_folds([(np.arange(len(groups)), np.arange(len(groups)))], groups)


def test_catboost_schema_is_stateless_and_excludes_ids():
    X = pd.DataFrame({c: [np.nan, 1] for c in NUMERIC_FEATURES})
    for c in CATEGORICAL_FEATURES:
        X[c] = [None, 'a']
    X['Num_Acc'], X['KSI'] = ['private', 'private'], [0, 1]
    prep = CatBoostFrame().fit(X)
    transformed = prep.transform(X)
    assert transformed.columns.tolist() == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert transformed[CATEGORICAL_FEATURES].iloc[0].eq('__MISSING__').all()
    assert transformed[NUMERIC_FEATURES].iloc[0].isna().all()


class NoSeenAccidents(ClassifierMixin, BaseEstimator):
    """Fail if a fitted accident is ever used for validation."""
    fit_history = []

    def fit(self, X, y):
        self.seen_ = set(X.accident)
        self.fit_history.append(self.seen_.copy())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        assert not self.seen_.intersection(X.accident)
        p = .2 + .5 * X.signal.to_numpy()
        return np.column_stack([1 - p, p])


def test_nested_stacking_excludes_outer_and_inner_accidents(tmp_path, monkeypatch):
    from src.models import ensembles, m4_common
    monkeypatch.setattr(ensembles, 'CACHE', tmp_path)
    monkeypatch.setattr(m4_common, 'CACHE', tmp_path)
    groups = pd.Series(np.repeat(np.arange(30), 2))
    y = pd.Series(np.tile([0, 1], 30))
    X = pd.DataFrame({'accident': groups, 'signal': y})
    folds = list(GroupKFold(5).split(X, y, groups))
    NoSeenAccidents.fit_history = []
    bases = {'guard': NoSeenAccidents()}
    base = {'guard': m4_common.oof_predict(bases['guard'], X, y, groups, folds, 'guard', 'test')}
    p = ensembles.nested_stacking_oof(bases, base, X, y, groups, folds, 'test')
    assert np.isfinite(p).all()
    assert len(p) == len(y)
    history = NoSeenAccidents.fit_history
    assert len(history) == 5 + 5 * 3
    for i, (train, valid) in enumerate(folds):
        for fitted_groups in history[5 + i * 3:5 + (i + 1) * 3]:
            assert fitted_groups.issubset(set(groups.iloc[train]))
            assert not fitted_groups.intersection(groups.iloc[valid])


def test_frozen_test_plan_rejects_changed_threshold():
    from src.evaluation.final_evaluation import check_frozen_plan
    check_frozen_plan({'threshold': .4}, {'threshold': .4})
    with pytest.raises(RuntimeError, match='already frozen'):
        check_frozen_plan({'threshold': .4}, {'threshold': .5})


def test_catboost_pipeline_clones_and_fits_real_estimator():
    from sklearn.base import clone
    from src.models.catboost_model import catboost_pipeline
    rng = np.random.default_rng(42)
    X = pd.DataFrame({name: rng.normal(size=40) for name in NUMERIC_FEATURES})
    for name in CATEGORICAL_FEATURES:
        X[name] = ['a', None, 'b', 'c'] * 10
    y = np.tile([0, 1], 20)
    model = clone(catboost_pipeline(iterations=3, depth=2))
    model.fit(X, y)
    assert model.predict_proba(X).shape == (40, 2)


def test_final_report_writes_expected_tables_and_figures(tmp_path, monkeypatch):
    from src.evaluation import final_evaluation as final
    monkeypatch.setattr(final, 'RESULTS_DIR', tmp_path)
    monkeypatch.setattr(final, 'FIGURES_DIR', tmp_path)
    monkeypatch.setattr(final.plt, 'show', lambda: None)
    y = pd.Series(np.tile([0, 1], 40))
    meta = pd.DataFrame({'Num_Acc': np.repeat(np.arange(40), 2), 'year': np.repeat([2024, 2025], 40)})
    p = np.linspace(.1, .9, len(y))
    predictions = {'Logistic baseline': p, 'Candidate': .15 + .7 * y.to_numpy()}
    summary, ci, paired, yearly = final.final_report(y, meta, predictions,
        {'Logistic baseline': .4, 'Candidate': .5}, n_boot=10)
    assert len(summary) == 2 and len(ci) == 14 and len(paired) == 7 and len(yearly) == 4
    assert (tmp_path / 'final_test.csv').exists()
    assert len(list(tmp_path.glob('*.png'))) == 3


def test_interpretation_and_error_analysis_interfaces(tmp_path, monkeypatch):
    from src.evaluation import m4_analysis as analysis
    from src.models.catboost_model import catboost_pipeline
    monkeypatch.setattr(analysis, 'RESULTS_DIR', tmp_path)
    monkeypatch.setattr(analysis, 'FIGURES_DIR', tmp_path)
    monkeypatch.setattr(analysis.plt, 'show', lambda: None)
    rng = np.random.default_rng(42)
    X = pd.DataFrame({name: rng.normal(size=200) for name in NUMERIC_FEATURES})
    for name in CATEGORICAL_FEATURES:
        X[name] = ['a', None, 'b', 'c'] * 50
    y = pd.Series(np.tile([0, 1], 100))
    groups = pd.Series(np.repeat(np.arange(100), 2))
    model = catboost_pipeline(iterations=3, depth=2).fit(X.iloc[:100], y.iloc[:100])
    meta = pd.DataFrame({'Num_Acc': groups, 'year': 2020, 'road_user': X.road_user,
                         'agg': X.agg, 'vma': 50, 'lum': X.lum, 'age': 35, 'obs': '2'})
    probabilities = model.predict_proba(X)[:, 1]
    importance, shap_table, risks, rates = analysis.interpret_catboost(
        model, X.iloc[100:], y.iloc[100:], groups.iloc[100:], meta, y, probabilities)
    assert len(importance) == len(X.columns)
    assert len(shap_table) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert risks.n_people.sum() == len(y)
    errors = analysis.error_analysis(meta, y, probabilities, .5, 'Synthetic', 'synthetic')
    assert set(errors['groups'].group) == {'road_user', 'agg', 'vma', 'lum', 'age_band', 'year'}
