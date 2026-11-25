# Owner: Sharan Raj JANAKIRAMAN (M3). Shared: every module scores models with this file.
"""One scoring function for the whole project, so all results are comparable.

Why not accuracy: only 18% of people are KSI. A model that always says
"not KSI" is 82% accurate and useless. So we use metrics about the KSI class:

  precision = TP / (TP + FP)  "when the model says KSI, how often is it right?"
  recall    = TP / (TP + FN)  "of all real KSI people, how many does it find?"
  F2        = 5 * P * R / (4 * P + R)
              general F-beta = (1 + b^2) P R / (b^2 P + R); with b = 2, recall
              counts 4 times (b^2) as much as precision, because missing a
              dangerous situation is worse than a false alarm (spec, objective A8).
  PR-AUC    = area under the precision-recall curve (average precision).
              Uses the probabilities, not one threshold. A random model scores
              the KSI rate (about 0.18), so 0.18 is the "no skill" level.
  ROC-AUC   = probability that a random KSI person gets a higher score than a
              random non-KSI person. 0.5 = no skill.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score, fbeta_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def evaluate(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Score predicted KSI probabilities.

    Input : y_true  = array of 0/1 (true KSI),
            y_proba = array of predicted probabilities of KSI (0 to 1),
            threshold = probability above which we say "KSI".
    Output: dict with pr_auc, roc_auc, precision, recall, f2, threshold,
            and the share of people predicted KSI.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "threshold": threshold,
        "predicted_ksi_share": y_pred.mean(),
    }


def best_f2_threshold(y_true, y_proba) -> float:
    """Threshold that maximises F2.

    Choose it on VALIDATION data only (inside cross-validation), never on the
    test years, otherwise the test score is optimistic.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    p, r = precision[:-1], recall[:-1]  # the last point has no threshold
    f2 = np.divide(5 * p * r, 4 * p + r, out=np.zeros_like(p), where=(4 * p + r) > 0)
    return float(thresholds[np.argmax(f2)])
