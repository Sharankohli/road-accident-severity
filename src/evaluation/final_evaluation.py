"""Freeze training choices, predict the held-out years once, then report cached scores."""
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits
from src.config import PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, TEST_YEARS
from src.features.build_features import add_features
from src.features.split import get_splits
from src.evaluation.metrics import evaluate
from src.evaluation.bootstrap import bootstrap_comparison
from src.models.m4_common import CACHE, fingerprint, environment
from src.models.ensembles import fit_stack, predict_stack


def _atomic_dump(value, path):
    tmp = path.with_suffix('.tmp')
    joblib.dump(value, tmp)
    tmp.replace(path)


def check_frozen_plan(existing, proposed):
    """A changed model/threshold/data after opening the test cannot silently rerun."""
    if existing != proposed:
        raise RuntimeError('The final test is already frozen under a different plan. Do not retune on test results.')


def final_predictions(ensemble, cat):
    run_id = fingerprint()
    if ensemble['run_id'] != run_id or cat['run_id'] != run_id:
        raise ValueError('Training artifacts do not match the current data/code/environment.')
    persons = add_features(pd.read_parquet(PROCESSED_DIR / 'persons.parquet'))
    X, y, groups, Xtest, ytest = get_splits(persons)
    if not np.array_equal(ensemble['index'], X.index) or not np.array_equal(cat['index'], X.index):
        raise ValueError('Training row order differs from the saved OOF artifacts.')
    testmeta = persons.loc[Xtest.index, ['Num_Acc', 'year']].copy()
    del persons
    chosen = ['Logistic baseline', ensemble['best_automl'], ensemble['best_ensemble'], 'CatBoost']
    thresholds = {name: ensemble['thresholds'][name] for name in chosen if name != 'CatBoost'}
    thresholds['CatBoost'] = cat['threshold']
    models = {name: ensemble['models'][name] for name in chosen if name not in ('CatBoost', 'Stacking')}
    models['CatBoost'] = cat['model']
    plan = {'run_id': run_id, 'chosen': chosen, 'thresholds': thresholds,
            'model_signature': joblib.hash({name: model.get_params() for name, model in models.items()}),
            'stack_signature': joblib.hash(ensemble['bases']) if 'Stacking' in chosen else None,
            'test_index_signature': joblib.hash(Xtest.index.to_numpy()),
            'test_years': TEST_YEARS, 'train_people': len(X), 'test_people': len(Xtest),
            'environment': environment()}
    plan_path = RESULTS_DIR / 'm4_final_plan.json'
    if plan_path.exists():
        check_frozen_plan(json.loads(plan_path.read_text())['plan'], plan)
    else:
        # Exclusive creation before any model sees test features or any score is computed.
        with plan_path.open('x') as f:
            json.dump({'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'plan': plan}, f, indent=2)
    prediction_path = CACHE / 'final_predictions.joblib'
    state = joblib.load(prediction_path) if prediction_path.exists() else {'plan': plan, 'predictions': {}}
    check_frozen_plan(state['plan'], plan)
    for name in chosen:
        if name in state['predictions']:
            print(f'{name}: reusing the frozen test predictions', flush=True)
            continue
        print(f'{name}: fitting on all training years, then predicting the test years once', flush=True)
        with threadpool_limits(limits=4):
            if name == 'Stacking':
                fitted, meta = fit_stack(ensemble['bases'], ensemble['base_oof'], X, y)
                p = predict_stack(fitted, meta, Xtest)
            else:
                fitted = clone(models[name]).fit(X, y)
                p = fitted.predict_proba(Xtest)[:, 1]
        if not np.isfinite(p).all() or len(p) != len(ytest):
            raise ValueError('Invalid final predictions.')
        state['predictions'][name] = p
        _atomic_dump(state, prediction_path)
        del fitted
    return ytest, testmeta, state['predictions'], thresholds


def final_report(y, meta, predictions, thresholds, n_boot=1000):
    ci, paired = bootstrap_comparison(y, predictions, meta.Num_Acc, thresholds,
        n_boot=n_boot, reference='Logistic baseline')
    ci.to_csv(RESULTS_DIR / 'final_test.csv', index=False)
    paired.to_csv(RESULTS_DIR / 'm4_final_paired.csv', index=False)
    summary = pd.DataFrame([{'model': name, **evaluate(y, p, thresholds[name])} for name, p in predictions.items()])
    summary.to_csv(RESULTS_DIR / 'm4_final_summary.csv', index=False)
    yearly = []
    for year in TEST_YEARS:
        mask = meta.year.to_numpy() == year
        for name, p in predictions.items():
            yearly.append({'year': year, 'model': name, 'n_people': int(mask.sum()),
                           'ksi_rate': np.asarray(y)[mask].mean(),
                           **evaluate(np.asarray(y)[mask], p[mask], thresholds[name])})
    yearly = pd.DataFrame(yearly)
    yearly.to_csv(RESULTS_DIR / 'm4_final_by_year.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, p in predictions.items():
        PrecisionRecallDisplay.from_predictions(y, p, name=name, ax=ax)
    ax.axhline(np.mean(y), linestyle='--', color='grey', label=f'KSI prevalence ({np.mean(y):.3f})')
    ax.set_title('Frozen models on 2024–2025: precision–recall')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'm4_final_pr_curves.png', dpi=180); plt.show(); plt.close(fig)
    ap = ci[ci.metric == 'pr_auc'].set_index('model').loc[list(predictions)].reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    # Percentile intervals need not contain the point estimate: draw endpoints directly.
    positions = np.arange(len(ap))
    ax.barh(positions, ap.estimate, alpha=.75)
    ax.hlines(positions, ap.low, ap.high, color='black', linewidth=2)
    ax.plot(ap.low, positions, '|', color='black'); ax.plot(ap.high, positions, '|', color='black')
    ax.set(yticks=positions, yticklabels=ap.model, xlabel='Average precision (PR-AUC)',
           title='Final test: 95% accident-bootstrap intervals')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'm4_final_pr_auc_ci.png', dpi=180); plt.show(); plt.close(fig)
    fig, axes = plt.subplots(len(predictions), 2, figsize=(10, 3.6 * len(predictions)))
    for row, (name, p) in enumerate(predictions.items()):
        for col, norm in enumerate([None, 'true']):
            ConfusionMatrixDisplay.from_predictions(y, p >= thresholds[name], normalize=norm,
                display_labels=['Not KSI', 'KSI'], ax=axes[row, col], colorbar=False,
                values_format='d' if norm is None else '.2f')
            axes[row, col].set_title(f'{name}: {"counts" if norm is None else "row proportions"}')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'm4_final_confusion.png', dpi=160); plt.show(); plt.close(fig)
    return summary, ci, paired, yearly
