"""OOF error analysis, held-out interpretation, and report figures for M4."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
from src.config import FIGURES_DIR, RESULTS_DIR, RANDOM_STATE


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'm4_{name}.png', dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def error_analysis(meta, y, p, threshold, model_name, prefix):
    frame = meta.copy()
    frame['actual'], frame['probability'] = np.asarray(y), np.asarray(p)
    frame['prediction'] = (np.asarray(p) >= threshold).astype(int)
    frame['age_band'] = pd.cut(frame.age, [0, 18, 30, 45, 60, 75, 101], right=False,
                               labels=['0–17', '18–29', '30–44', '45–59', '60–74', '75–100'])
    rows = []
    for column in ['road_user', 'agg', 'vma', 'lum', 'age_band', 'year']:
        for value, part in frame.groupby(frame[column].astype('string').fillna('Missing'), sort=True):
            actual, pred = part.actual.to_numpy(), part.prediction.to_numpy()
            tp = int(((actual == 1) & (pred == 1)).sum())
            fn = int(((actual == 1) & (pred == 0)).sum())
            fp = int(((actual == 0) & (pred == 1)).sum())
            rows.append(dict(group=column, value=value, n_people=len(part), n_ksi=int(actual.sum()),
                             tp=tp, fn=fn, fp=fp, recall=tp/(tp+fn) if tp+fn else np.nan,
                             precision=tp/(tp+fp) if tp+fp else np.nan,
                             mean_prediction=part.probability.mean(), observed_ksi=part.actual.mean()))
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / f'm4_{prefix}_group_errors.csv', index=False)
    fn = frame[(frame.actual == 1) & (frame.prediction == 0)]
    fn_groups = fn.groupby(['road_user', 'agg', 'vma'], dropna=False).agg(
        missed_people=('actual', 'size'), mean_score=('probability', 'mean')).reset_index().sort_values('missed_people', ascending=False)
    fn_groups.to_csv(RESULTS_DIR / f'm4_{prefix}_false_negatives.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, normalize, title in zip(axes, [None, 'true'], ['People', 'Share within actual class']):
        ConfusionMatrixDisplay.from_predictions(y, frame.prediction, normalize=normalize,
            display_labels=['Not KSI', 'KSI'], values_format='d' if normalize is None else '.2f', ax=ax, colorbar=False)
        ax.set_title(title)
    fig.suptitle(f'{model_name}: training OOF confusion matrices')
    _save(fig, f'{prefix}_confusion')
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y, p, n_bins=10, strategy='quantile', ax=ax, name=model_name)
    ax.set_title('Training OOF calibration (equal-count bins)')
    _save(fig, f'{prefix}_calibration')
    eligible = table[(table.n_ksi >= 30) & (table.n_people >= 100)].sort_values('recall')
    worst = eligible.iloc[0] if len(eligible) else None
    summary = f'At threshold {threshold:.4f}, we missed {len(fn):,} of {int(np.sum(y)):,} KSI people. '
    summary += f'The Brier score is {brier_score_loss(y, p):.4f}. '
    if worst is not None:
        summary += f'Among groups with at least 100 people and 30 KSI cases, the lowest recall is {worst.recall:.1%} for {worst.group}={worst.value}. '
    summary += 'The table shows where misses happen; it cannot tell us which missing measurement caused a miss. Actual impact speed, distraction and emergency response detail could help, but are not established explanations here.'
    return {'groups': table, 'false_negatives': fn_groups, 'summary': summary}


def interpret_catboost(fitted, X_valid, y_valid, groups_valid, meta, y_all, p_all):
    from catboost import Pool
    from sklearn.inspection import permutation_importance
    from src.models.m4_common import accident_indices
    from src.features.preprocess import CATEGORICAL_FEATURES
    import shap
    ix = accident_indices(groups_valid, max_people=3000)
    xs, ys = X_valid.iloc[ix], y_valid.iloc[ix]
    # Evaluation rows are complete accidents never seen by this fitted fold model.
    result = permutation_importance(fitted, xs, ys, scoring='average_precision',
        n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({'feature': xs.columns, 'mean_ap_drop': result.importances_mean,
                                'std_ap_drop': result.importances_std}).sort_values('mean_ap_drop', ascending=False)
    importance.to_csv(RESULTS_DIR / 'm4_permutation_importance.csv', index=False)
    top = importance.head(15).sort_values('mean_ap_drop')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.feature, top.mean_ap_drop, xerr=top.std_ap_drop, capsize=3)
    ax.set(xlabel='Drop in average precision after permutation', title='CatBoost: held-out training accidents (3 repeats)')
    _save(fig, 'permutation_importance')
    data = fitted.named_steps['prep'].transform(xs)
    model = fitted.named_steps['model']
    values = model.get_feature_importance(Pool(data, cat_features=CATEGORICAL_FEATURES), type='ShapValues')
    raw = model.predict(data, prediction_type='RawFormulaVal')
    np.testing.assert_allclose(values[:, :-1].sum(axis=1) + values[:, -1], raw, atol=1e-6)
    shap_table = pd.DataFrame({'feature': data.columns, 'mean_abs_shap': np.abs(values[:, :-1]).mean(axis=0)})
    shap_table = shap_table.sort_values('mean_abs_shap', ascending=False)
    shap_table.to_csv(RESULTS_DIR / 'm4_shap_importance.csv', index=False)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(values[:, :-1], feature_names=data.columns, plot_type='bar', max_display=15, show=False)
    plt.title('CatBoost: average absolute SHAP contribution (log-odds)')
    _save(plt.gcf(), 'shap_summary')
    risks = meta[['road_user', 'vma', 'obs']].copy()
    risks['predicted_score'], risks['actual'] = np.asarray(p_all), np.asarray(y_all)
    risks = risks.groupby(['road_user', 'vma', 'obs'], dropna=False).agg(
        n_people=('actual', 'size'), n_ksi=('actual', 'sum'), observed_ksi_rate=('actual', 'mean'),
        predicted_ksi_score=('predicted_score', 'mean')).reset_index()
    risks['eligible_for_ranking'] = risks.n_people >= 100
    risks = risks.sort_values(['eligible_for_ranking', 'predicted_ksi_score'], ascending=False)
    risks.to_csv(RESULTS_DIR / 'm4_risk_situations.csv', index=False)
    eda = meta.copy()
    eda['actual'] = np.asarray(y_all)
    summaries = []
    for feature in ['road_user', 'vma', 'obs']:
        for value, part in eda.groupby(feature, dropna=False):
            summaries.append({'feature': feature, 'value': value, 'n_people': len(part), 'ksi_rate': part.actual.mean()})
    eda_table = pd.DataFrame(summaries)
    eda_table.to_csv(RESULTS_DIR / 'm4_training_risk_rates.csv', index=False)
    return importance, shap_table, risks, eda_table
