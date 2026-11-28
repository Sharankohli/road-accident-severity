"""Accident-cluster percentile bootstrap with exact weighted metric evaluation.

Repeated accidents are represented by integer row weights. Sorting predictions once
makes 1,000 draws practical; tests compare this with physically duplicated rows
passed to the project's evaluate(). Models and thresholds remain fixed.
"""
from numbers import Integral
import numpy as np
import pandas as pd
from src.config import RANDOM_STATE
from src.evaluation.metrics import evaluate


def _validate(y, p, groups, threshold):
    series = [v for v in (y, p, groups) if isinstance(v, pd.Series)]
    if series and any(not s.index.equals(series[0].index) for s in series[1:]):
        raise ValueError('Series indexes must match in the same order.')
    y, p, groups = np.asarray(y), np.asarray(p, dtype=float), np.asarray(groups)
    if any(v.ndim != 1 for v in (y, p, groups)) or not (len(y) == len(p) == len(groups)) or not len(y):
        raise ValueError('Inputs must be nonempty, aligned 1D arrays.')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both binary classes are required.')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and between zero and one.')
    if pd.isna(groups).any() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Groups cannot be missing; threshold must be in [0, 1].')
    return y.astype(int), p, groups


class _WeightedMetrics:
    def __init__(self, y, p, threshold):
        self.y, self.p, self.threshold = y, p, threshold
        self.values, self.bins = np.unique(p, return_inverse=True)
        self.positive = y == 1
        self.predicted = p >= threshold

    def __call__(self, weights):
        pos = np.bincount(self.bins, weights=weights * self.positive, minlength=len(self.values))
        neg = np.bincount(self.bins, weights=weights * ~self.positive, minlength=len(self.values))
        positives, negatives = pos.sum(), neg.sum()
        if positives == 0 or negatives == 0:
            return None
        tp_curve = np.cumsum(pos[::-1])
        fp_curve = np.cumsum(neg[::-1])
        precision_curve = np.divide(tp_curve, tp_curve + fp_curve,
                                    out=np.zeros_like(tp_curve), where=(tp_curve + fp_curve) > 0)
        ap = np.sum(pos[::-1] * precision_curve) / positives
        auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (positives * negatives)
        tp = np.sum(weights * self.positive * self.predicted)
        predicted = np.sum(weights * self.predicted)
        precision = tp / predicted if predicted else 0.
        recall = tp / positives
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.
        return dict(pr_auc=ap, roc_auc=auc, precision=precision, recall=recall, f2=f2,
                    threshold=self.threshold, predicted_ksi_share=predicted / (positives + negatives))


def bootstrap_comparison(y_true, predictions, groups, thresholds=None, n_boot=1000,
                         alpha=0.05, reference=None, seed=RANDOM_STATE):
    """Return model CIs and paired model-minus-reference CIs from identical draws.

    Sampling uncertainty is conditional on these fitted predictions; it does not
    include retraining or hyperparameter selection uncertainty. Single-class
    resamples are redrawn and their count is stored in DataFrame.attrs.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 2:
        raise ValueError('n_boot must be an integer of at least two.')
    if not 0 < alpha < 1 or not predictions:
        raise ValueError('Require predictions and 0 < alpha < 1.')
    if reference is not None and reference not in predictions:
        raise ValueError('Reference model is missing.')
    thresholds = thresholds or {name: 0.5 for name in predictions}
    scorers, estimates = {}, {}
    for name, p in predictions.items():
        y, probs, g = _validate(y_true, p, groups, thresholds[name])
        scorers[name] = _WeightedMetrics(y, probs, thresholds[name])
        estimates[name] = evaluate(y, probs, thresholds[name])
    codes, ids = pd.factorize(g, sort=False)
    if len(ids) < 2:
        raise ValueError('At least two accidents are required.')
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in predictions}
    rejected = 0
    while len(next(iter(samples.values()))) < n_boot:
        multiplicity = np.bincount(rng.integers(0, len(ids), size=len(ids)), minlength=len(ids))
        weights = multiplicity[codes]
        scores = {name: scorer(weights) for name, scorer in scorers.items()}
        if any(v is None for v in scores.values()):
            rejected += 1
            if rejected > 10 * n_boot:
                raise ValueError('Too many single-class resamples for a reliable interval.')
            continue
        for name, score in scores.items():
            samples[name].append(score)
    frames = {name: pd.DataFrame(rows) for name, rows in samples.items()}
    rows, differences = [], []
    for name, frame in frames.items():
        for metric in frame.columns:
            low, high = frame[metric].quantile([alpha / 2, 1 - alpha / 2])
            rows.append(dict(model=name, metric=metric, estimate=estimates[name][metric], low=low, high=high))
            if reference is not None and name != reference:
                delta = frame[metric] - frames[reference][metric]
                low, high = delta.quantile([alpha / 2, 1 - alpha / 2])
                differences.append(dict(model=name, reference=reference, metric=metric,
                                        estimate=estimates[name][metric] - estimates[reference][metric],
                                        low=low, high=high))
    ci, paired = pd.DataFrame(rows), pd.DataFrame(differences)
    for table in (ci, paired):
        table.attrs.update(n_boot=n_boot, rejected_single_class=rejected, n_accidents=len(ids), seed=seed)
    return ci, paired


def bootstrap_ci(y_true, y_proba, groups, n_boot=1000, threshold=0.5, alpha=0.05):
    ci, _ = bootstrap_comparison(y_true, {'model': y_proba}, groups,
                                 {'model': threshold}, n_boot=n_boot, alpha=alpha)
    return ci.drop(columns='model').set_index('metric')
