# Owner: Sharan Raj JANAKIRAMAN (M3)
"""Error analysis of the best AutoML v0.1 model (Week 3 objective).

Run from the repo root:
    python -u -m src.models.error_analysis

Takes the best logistic-regression setting from the grid search, gets
out-of-fold predictions on ALL training years (each person is predicted by a
model that never saw their accident), picks the best-F2 threshold on these
out-of-fold predictions (never on the test years), then measures where the
model fails: recall (share of KSI people found) and precision by group.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from src.evaluation.metrics import best_f2_threshold, evaluate
from src.features.split import get_cv
from src.models.automl import BEST_JSON, SEARCH_SPACE, load_training, make_pipeline
from src.config import RESULTS_DIR

MODEL = "Logistic regression"
GROUP_COLS = ["road_user", "agg", "lum", "catr", "age_band"]
MIN_KSI = 300  # groups with fewer KSI people are too small to read


def main() -> None:
    X, y, groups = load_training()
    best = json.loads(BEST_JSON.read_text(encoding="utf-8"))[MODEL]
    kind, estimator, _, _ = SEARCH_SPACE[MODEL]
    pipe = make_pipeline(kind, estimator).set_params(**best["params"])
    t = time.time()
    folds = list(get_cv().split(X, y, groups))
    oof = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba", n_jobs=3)[:, 1]
    print(f"out-of-fold predictions: {time.time() - t:.0f} s", flush=True)
    np.save(RESULTS_DIR / "oof_logistic_regression.npy", oof.astype(np.float32))

    thr = best_f2_threshold(y, oof)
    overall = {"model": f"{MODEL} {best['params']}", **evaluate(y, oof, threshold=thr)}
    pred = (oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    overall.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    pd.DataFrame([overall]).round(4).to_csv(RESULTS_DIR / "error_overall.csv", index=False)
    print(overall, flush=True)

    df = X.copy()
    df["age_band"] = pd.cut(df["age"], [0, 17, 24, 44, 64, 74, 120],
                            labels=["0-17", "18-24", "25-44", "45-64", "65-74", "75+"])
    df["y"], df["pred"] = y.values, pred
    rows = []
    for col in GROUP_COLS:
        for val, g in df.groupby(col, observed=True, dropna=False):
            ksi = int(g.y.sum())
            if ksi < MIN_KSI:
                continue
            flagged = int(g.pred.sum())
            rows.append({"variable": col, "group": str(val), "people": len(g), "ksi_people": ksi,
                         "ksi_rate": g.y.mean(), "recall": g.pred[g.y == 1].mean(),
                         "precision": g.y[g.pred == 1].mean() if flagged else np.nan,
                         "missed_ksi": int(((g.y == 1) & (g.pred == 0)).sum()),
                         "false_alarms": int(((g.y == 0) & (g.pred == 1)).sum())})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(RESULTS_DIR / "error_by_group.csv", index=False)
    print(out.to_string(), flush=True)


if __name__ == "__main__":
    main()
