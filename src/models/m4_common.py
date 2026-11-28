"""Shared, resumable training-only machinery for the M4 notebooks."""
import hashlib
import json
import platform
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src.config import ROOT, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR
from src.features.build_features import add_features
from src.features.split import get_cv, get_splits
from src.evaluation.metrics import best_f2_threshold, evaluate

CACHE = ROOT / 'data' / 'm4'
CACHE.mkdir(parents=True, exist_ok=True)


def fingerprint():
    """Invalidate caches when data, fitted code, or package versions change."""
    h = hashlib.sha256()
    path = PROCESSED_DIR / 'persons.parquet'
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    for folder in ('features', 'models'):
        for path in sorted((ROOT / 'src' / folder).glob('*.py')):
            h.update(path.read_bytes())
    h.update((RESULTS_DIR / 'automl_v01_best_params.json').read_bytes())
    h.update(json.dumps(environment(), sort_keys=True).encode())
    return h.hexdigest()[:16]


def environment():
    return {'python': platform.python_version(), **{p: version(p) for p in
            ('numpy', 'pandas', 'scikit-learn', 'catboost', 'shap')}}


def load_training():
    persons = add_features(pd.read_parquet(PROCESSED_DIR / 'persons.parquet'))
    X, y, groups, _, _ = get_splits(persons)
    meta = persons.loc[X.index, ['Num_Acc', 'year', 'road_user', 'agg', 'vma', 'lum', 'age', 'obs']].copy()
    del persons
    return X, y, groups, meta


def accident_indices(groups, max_people=100_000, seed=RANDOM_STATE):
    """Choose whole accidents without replacement, up to a person budget."""
    codes, ids = pd.factorize(np.asarray(groups), sort=False)
    sizes = np.bincount(codes)
    order = np.random.default_rng(seed).permutation(len(ids))
    keep = order[np.cumsum(sizes[order]) <= max_people]
    if not len(keep):
        raise ValueError('Budget is too small for one complete accident.')
    return np.flatnonzero(np.isin(codes, keep))


def grouped_folds(X, y, groups, key):
    path = CACHE / f'{key}_folds.joblib'
    signature = joblib.hash((X.index.to_numpy(), np.asarray(y), np.asarray(groups), repr(get_cv())))
    if path.exists():
        saved = joblib.load(path)
        if saved['signature'] == signature:
            return saved['folds']
    folds = list(get_cv().split(X, y, groups))
    validate_folds(folds, groups)
    joblib.dump({'signature': signature, 'folds': folds}, path)
    return folds


def validate_folds(folds, groups):
    groups = np.asarray(groups)
    seen = np.zeros(len(groups), int)
    for train, valid in folds:
        if set(groups[train]) & set(groups[valid]):
            raise ValueError('Accidents overlap between training and validation.')
        seen[valid] += 1
    if not np.all(seen == 1):
        raise ValueError('Every row must be validated exactly once.')


def oof_predict(model, X, y, groups, folds, key, run_id):
    """Save each completed fold; never reuse predictions from another configuration."""
    validate_folds(folds, groups)
    signature = joblib.hash((run_id, model.get_params(), X.index.to_numpy(), np.asarray(y), folds))
    out = np.full(len(y), np.nan)
    for i, (train, valid) in enumerate(folds):
        path = CACHE / f'{key}_fold{i}.joblib'
        saved = joblib.load(path) if path.exists() else {}
        if saved.get('signature') == signature:
            out[valid] = saved['prediction']
            print(f'{key}: fold {i + 1}/{len(folds)} loaded', flush=True)
            continue
        start = time.monotonic()
        with threadpool_limits(limits=4):
            fitted = clone(model).fit(X.iloc[train], y.iloc[train])
            p = fitted.predict_proba(X.iloc[valid])[:, 1]
        out[valid] = p
        joblib.dump({'signature': signature, 'prediction': p}, path)
        # Fold 0 is also a genuinely held-out interpretation model.
        if i == 0:
            joblib.dump({'signature': signature, 'model': fitted, 'valid': valid}, CACHE / f'{key}_interpret.joblib')
        print(f'{key}: fold {i + 1}/{len(folds)} finished in {time.monotonic() - start:.1f}s', flush=True)
    if not np.isfinite(out).all():
        raise ValueError('Incomplete OOF probabilities.')
    return out


def score_oof(name, y, probabilities):
    threshold = best_f2_threshold(y, probabilities)
    return {'model': name, 'scored_on': 'out-of-fold (5 grouped folds)',
            **evaluate(y, probabilities, threshold)}


def tune_candidates(candidates, X, y, groups, key, run_id, max_people=100_000):
    ix = accident_indices(groups, max_people)
    xs, ys, gs = X.iloc[ix], y.iloc[ix], groups.iloc[ix]
    folds = grouped_folds(xs, ys, gs, f'tune_{max_people}')
    rows = []
    for name, model in candidates.items():
        p = oof_predict(model, xs, ys, gs, folds, f'{key}_{name}', run_id)
        rows.append(score_oof(name, ys, p))
    table = pd.DataFrame(rows).sort_values('pr_auc', ascending=False)
    table['n_people'] = len(ix)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / f'm4_{key}_search.csv', index=False)
    best = table.iloc[0]['model']
    return clone(candidates[best]), table
