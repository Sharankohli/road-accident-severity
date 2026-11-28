"""M3 handoff, bagging, boosting, voting, and nested accident-grouped stacking."""
import json
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from src.config import RANDOM_STATE, RESULTS_DIR
from src.models.automl import SEARCH_SPACE, FULL_DATA_MODELS, make_pipeline
from src.models.m4_common import CACHE, oof_predict, validate_folds


def automl_models():
    best = json.loads((RESULTS_DIR / 'automl_v01_best_params.json').read_text())
    return {name: make_pipeline(SEARCH_SPACE[name][0], clone(SEARCH_SPACE[name][1])).set_params(**best[name]['params'])
            for name in FULL_DATA_MODELS}


def ensemble_candidates():
    tree = automl_models()['Decision tree'].named_steps['model']
    bagging = {}
    for n, fraction in [(20, 0.5), (40, 0.5), (40, 0.8)]:
        bagging[f'bag_{n}_{fraction}'] = make_pipeline('tree', BaggingClassifier(
            estimator=clone(tree), n_estimators=n, max_samples=fraction,
            random_state=RANDOM_STATE, n_jobs=4))
    boosting = {}
    for leaves, rate, iterations in [(15, 0.1, 150), (31, 0.1, 150), (31, 0.05, 250)]:
        pipe = make_pipeline('tree', HistGradientBoostingClassifier(
            max_leaf_nodes=leaves, learning_rate=rate, max_iter=iterations,
            class_weight='balanced', early_stopping=False, random_state=RANDOM_STATE))
        # No random internal early-stopping split that could separate an accident.
        pipe.named_steps['prep'].set_params(sparse_threshold=0)
        boosting[f'hgb_{leaves}_{rate}_{iterations}'] = pipe
    return bagging, boosting


def voting_model(bases):
    return VotingClassifier(estimators=[(name, clone(model)) for name, model in bases.items()], voting='soft', n_jobs=1)


def nested_stacking_oof(bases, base_oof, X, y, groups, folds, run_id):
    """Outer 5-fold evaluation; inner 3-fold grouped predictions train the meta-model.

    A global OOF table must NOT be split again to evaluate the meta-model: its
    training features could have been learned using the outer validation labels.
    The inner models below never see the outer validation accidents.
    """
    validate_folds(folds, groups)
    names = list(bases)
    output = np.full(len(y), np.nan)
    for i, (train, valid) in enumerate(folds):
        xs, ys, gs = X.iloc[train], y.iloc[train], groups.iloc[train]
        path = CACHE / f'stack_inner{i}_folds.joblib'
        signature = joblib.hash((xs.index, ys, gs, RANDOM_STATE))
        saved = joblib.load(path) if path.exists() else {}
        if saved.get('signature') == signature:
            inner = saved['folds']
        else:
            inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE).split(xs, ys, gs))
            joblib.dump({'signature': signature, 'folds': inner}, path)
        inner_probs = np.column_stack([oof_predict(bases[name], xs, ys, gs, inner,
                                      f'stack_outer{i}_{name}', run_id) for name in names])
        meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        meta.fit(inner_probs, ys)
        # These predictions come from base models fitted only on this outer train.
        valid_probs = np.column_stack([base_oof[name][valid] for name in names])
        output[valid] = meta.predict_proba(valid_probs)[:, 1]
        print(f'Stacking outer fold {i + 1}/5 finished', flush=True)
    return output


def fit_stack(bases, base_oof, X, y):
    """Final training: meta-model learns from OOF probabilities, bases from all train."""
    names = list(bases)
    meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(
        np.column_stack([base_oof[name] for name in names]), y)
    fitted = {name: clone(model).fit(X, y) for name, model in bases.items()}
    return fitted, meta


def predict_stack(fitted, meta, X):
    return meta.predict_proba(np.column_stack([model.predict_proba(X)[:, 1] for model in fitted.values()]))[:, 1]


def compare_on_subsample(models, X, y, groups, run_id, n_people=30_000):
    """The shared 30k benchmark requested in SONG_ENSEMBLES_TODAY.md."""
    import pandas as pd
    from sklearn.model_selection import cross_validate
    from threadpoolctl import threadpool_limits
    from src.models.automl import accident_subsample
    from src.models.m4_common import grouped_folds
    xs, ys, gs = accident_subsample(X, y, groups, n_people)
    folds = grouped_folds(xs, ys, gs, f'comparison_{n_people}')
    rows = []
    for name, model in models.items():
        path = CACHE / f'comparison_{name}.joblib'
        signature = joblib.hash((run_id, model.get_params(), xs.index, ys, folds))
        saved = joblib.load(path) if path.exists() else {}
        if saved.get('signature') == signature:
            scores = saved['scores']
        else:
            with threadpool_limits(limits=4):
                scores = cross_validate(model, xs, ys, cv=folds, scoring='average_precision',
                                        return_train_score=True, n_jobs=1, error_score='raise')
            joblib.dump({'signature': signature, 'scores': scores}, path)
        rows.append({'model': name, 'cv_pr_auc_mean': scores['test_score'].mean(),
                     'cv_pr_auc_std': scores['test_score'].std(ddof=0),
                     'train_pr_auc_mean': scores['train_score'].mean(),
                     'subsample_people': len(xs), 'fit_time_s': scores['fit_time'].mean()})
        print(f'{name}: validation AP {rows[-1]["cv_pr_auc_mean"]:.4f}', flush=True)
    return pd.DataFrame(rows).sort_values('cv_pr_auc_mean', ascending=False)
