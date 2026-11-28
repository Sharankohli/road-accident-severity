# Owner: Sharan Raj JANAKIRAMAN (M3)
"""AutoML v0.2: corrected grid search (fixes the limits of v0.1).

Run from the repo root (training years 2019-2023 ONLY; test years never used):
    python -u -m src.models.automl_v2 search   # tune each model
    python -u -m src.models.automl_v2 equal    # all tuned models on ONE common sample
    python -u -m src.models.automl_v2 full     # best LR / tree / forest, out-of-fold on all training years

What v0.1 got wrong and how v0.2 fixes it:
  1. Unequal samples (LR 100k, trees 30k, kNN 20k, SVM 10k) made the model
     ranking unfair. -> LR, tree and forest are tuned on the SAME 100,000
     people; kNN and SVM (which scale badly) on the SAME 20,000 people, and
     step "equal" scores all five tuned models on one identical 20,000 sample.
  2. Trees were slow because they received a sparse matrix. -> dense input
     for tree models (sparse_threshold=0), so bigger grids fit in time.
  3. Grids were tiny. -> wider grids (deeper trees, more leaf sizes, more C).
  4. SVM had an iteration cap. -> no cap (max_iter=-1): every fit converges.

v0.1 files are kept unchanged because the M4 ensembles and the frozen final
test were built on them; v0.2 is a training-years analysis and does not
replace any model already scored on 2024-2025.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE, RESULTS_DIR
from src.evaluation.metrics import best_f2_threshold, evaluate
from src.features.split import get_cv
from src.models.automl import accident_subsample, load_training, make_pipeline

BIG, SMALL = 100_000, 20_000
SEARCH_CSV = RESULTS_DIR / "automl_v02_search.csv"
BEST_JSON = RESULTS_DIR / "automl_v02_best_params.json"

# name -> (preprocessor kind, dense?, estimator, grid, sample size, parallel jobs)
SEARCH_SPACE = {
    "Logistic regression": (
        "linear", False,
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
        {"model__C": [0.001, 0.01, 0.1, 1, 10]}, BIG, 4),
    "Decision tree": (
        "tree", True,
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"model__max_depth": [6, 10, 14, 18, 24, None],
         "model__min_samples_leaf": [20, 50, 100, 200]}, BIG, 4),
    "Random forest": (
        "tree", True,
        RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", max_features="sqrt",
                               n_jobs=2, random_state=RANDOM_STATE),
        {"model__max_depth": [20, None],
         "model__min_samples_leaf": [1, 5, 20]}, BIG, 2),
    "k-nearest neighbours": (
        "linear", False,
        KNeighborsClassifier(),
        {"model__n_neighbors": [25, 51, 101, 201],
         "model__weights": ["uniform", "distance"]}, SMALL, 4),
    "SVM": (
        "linear", False,
        SVC(kernel="rbf", gamma="scale", class_weight="balanced", max_iter=-1, random_state=RANDOM_STATE),
        {"model__C": [0.3, 1, 3]}, SMALL, 3),
}
FULL_DATA_MODELS = ["Logistic regression", "Decision tree", "Random forest"]


def build(name):
    kind, dense, estimator, _, _, _ = SEARCH_SPACE[name]
    pipe = make_pipeline(kind, clone(estimator))
    if dense:  # trees are much faster on a dense matrix
        pipe.named_steps["prep"].set_params(sparse_threshold=0)
    return pipe


def _save(rows, best):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(SEARCH_CSV, index=False)
    BEST_JSON.write_text(json.dumps(best, indent=1, default=float), encoding="utf-8")


def run_search():
    X, y, groups = load_training()
    rows, best = [], {}
    if BEST_JSON.exists() and "--fresh" not in sys.argv:  # resume after a crash
        best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
        rows = pd.read_csv(SEARCH_CSV).to_dict("records")
    samples = {}
    for name, (_, _, _, grid, n_people, jobs) in SEARCH_SPACE.items():
        if name in best:
            print(f"skip {name} (already done)", flush=True)
            continue
        if n_people not in samples:  # one shared sample per size: identical people for every model
            Xs, ys, gs = accident_subsample(X, y, groups, n_people)
            samples[n_people] = (Xs, ys, list(get_cv().split(Xs, ys, gs)))
        Xs, ys, folds = samples[n_people]
        print(f"[{time.strftime('%H:%M:%S')}] start {name} ({len(Xs):,} people)", flush=True)
        search = GridSearchCV(build(name), grid, scoring="average_precision", cv=folds,
                              n_jobs=jobs, refit=False, return_train_score=True)
        t = time.time()
        search.fit(Xs, ys)
        secs = time.time() - t
        res = pd.DataFrame(search.cv_results_)
        for _, r in res.iterrows():
            rows.append({"model": name, "params": json.dumps({k.replace("model__", ""): v for k, v in r.params.items()}),
                         "cv_pr_auc_mean": r.mean_test_score, "cv_pr_auc_std": r.std_test_score,
                         "train_pr_auc_mean": r.mean_train_score, "fit_time_s": r.mean_fit_time,
                         "subsample_people": len(Xs)})
        i = res.mean_test_score.idxmax()
        best[name] = {"params": res.params[i], "cv_pr_auc_mean": res.mean_test_score[i],
                      "cv_pr_auc_std": res.std_test_score[i], "train_pr_auc_mean": res.mean_train_score[i],
                      "subsample_people": len(Xs), "search_time_s": round(secs)}
        print(f"{name:22} best PR-AUC {res.mean_test_score[i]:.4f} +/- {res.std_test_score[i]:.4f} "
              f"(train {res.mean_train_score[i]:.4f}) {res.params[i]}  [{secs:.0f} s]", flush=True)
        _save(rows, best)


def run_equal():
    """Every tuned model on the SAME 20,000 people and folds: a fair ranking."""
    X, y, groups = load_training()
    best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    Xs, ys, gs = accident_subsample(X, y, groups, SMALL)
    folds = list(get_cv().split(Xs, ys, gs))
    rows = []
    for name in SEARCH_SPACE:
        t = time.time()
        s = cross_validate(build(name).set_params(**best[name]["params"]), Xs, ys, cv=folds,
                           scoring="average_precision", return_train_score=True, n_jobs=3)
        rows.append({"model": name, "cv_pr_auc_mean": s["test_score"].mean(), "cv_pr_auc_std": s["test_score"].std(),
                     "train_pr_auc_mean": s["train_score"].mean(), "people": len(Xs)})
        print(f"{name:22} equal-sample PR-AUC {rows[-1]['cv_pr_auc_mean']:.4f} ({time.time() - t:.0f} s)", flush=True)
    pd.DataFrame(rows).round(4).to_csv(RESULTS_DIR / "automl_v02_equal_sample.csv", index=False)


def run_full():
    """Best LR / tree / forest, out-of-fold on all 619,552 training people."""
    X, y, groups = load_training()
    best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    folds = list(get_cv().split(X, y, groups))
    out = RESULTS_DIR / "automl_v02_full_train.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["model"] for r in rows}
    for name in FULL_DATA_MODELS:
        if name in done:
            continue
        t = time.time()
        jobs = 1 if name == "Random forest" else 3  # the forest already uses 2 cores and a lot of RAM
        oof = cross_val_predict(build(name).set_params(**best[name]["params"]), X, y, cv=folds,
                                method="predict_proba", n_jobs=jobs)[:, 1]
        thr = best_f2_threshold(y, oof)
        rows.append({"model": name, "threshold_rule": "best F2", **evaluate(y, oof, threshold=thr),
                     "time_s": round(time.time() - t)})
        print(f"{name:22} out-of-fold PR-AUC {rows[-1]['pr_auc']:.4f} ({time.time() - t:.0f} s)", flush=True)
        pd.DataFrame(rows).round(4).to_csv(out, index=False)


if __name__ == "__main__":
    {"search": run_search, "equal": run_equal, "full": run_full}[sys.argv[1]]()
