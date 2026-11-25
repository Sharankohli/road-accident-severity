# Owner: Sharan Raj JANAKIRAMAN (M3)
"""Score the three baselines on the TRAINING years (2019-2023) only.

Run from the repo root:   python -m src.models.run_baselines
Outputs:
  results/baseline_train.csv        metrics of the 3 baselines
  results/logreg_odds_ratios.csv    odds ratios of the logistic regression

How each baseline is scored (test years 2024-2025 are never used here):
  - Majority class and expert rule learn nothing (the rule is hand-written),
    so they are scored directly on all training years.
  - Logistic regression learns weights, so it is scored with OUT-OF-FOLD
    predictions: M2's 5 accident-grouped folds; each person is predicted by a
    model that never saw their accident. Encoding/scaling are refitted inside
    each fold (Pipeline), so there is no leakage.
  - Threshold: 0.5, plus the threshold that maximises F2 on these out-of-fold
    predictions (a validation choice; it will be re-checked on the test years).
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.evaluation.metrics import best_f2_threshold, evaluate
from src.features.build_features import add_features
from src.features.split import get_cv, get_splits
from src.models.baseline import ExpertRuleBaseline, logistic_baseline, majority_baseline


def odds_ratios(pipeline) -> pd.DataFrame:
    """Odds ratio per transformed feature = exp(coefficient).

    > 1: associated with higher KSI odds; < 1: lower. Numeric features are
    standardised, so their odds ratio is per 1 standard deviation.
    """
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    return table.sort_values("coef", ascending=False).reset_index(drop=True)


def main() -> None:
    persons = pd.read_parquet(PROCESSED_DIR / "persons.parquet")
    X, y, groups, _, _ = get_splits(add_features(persons))  # test part ignored on purpose
    print(f"Training years: {len(X):,} people, KSI rate {y.mean():.1%}")

    rows = []
    for name, model in [("Majority class", majority_baseline()), ("Expert rule", ExpertRuleBaseline())]:
        model.fit(X, y)
        rows.append({"model": name, "scored_on": "all training years", **evaluate(y, model.predict_proba(X)[:, 1])})

    t = time.time()
    folds = list(get_cv().split(X, y, groups))  # computed once, reused below
    oof = cross_val_predict(logistic_baseline(), X, y, cv=folds, method="predict_proba", n_jobs=-1)[:, 1]
    print(f"Logistic regression cross-validation: {time.time() - t:.0f} s")
    rows.append({"model": "Logistic regression", "scored_on": "out-of-fold (5 grouped folds)", **evaluate(y, oof)})
    thr = best_f2_threshold(y, oof)
    rows.append({"model": "Logistic regression (F2 threshold)", "scored_on": "out-of-fold (5 grouped folds)",
                 **evaluate(y, oof, threshold=thr)})

    results = pd.DataFrame(rows).set_index("model").round(4)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "baseline_train.csv")
    print(results.drop(columns="scored_on").to_string())

    final = logistic_baseline().fit(X, y)  # all training years, for interpretation only
    odds_ratios(final).round(4).to_csv(RESULTS_DIR / "logreg_odds_ratios.csv", index=False)


if __name__ == "__main__":
    main()
