# Owner: Sharan Raj JANAKIRAMAN (M3)
"""AutoML v0.1: grid search over standard course models (LAB 6 Part II).

Run from the repo root:
    python -u -m src.models.automl search   # step 1: grid search on a subsample (about 30-60 min)
    python -m src.models.automl full     # step 2: best settings, out-of-fold on all training years

Design (each point avoids a LAB 6 trap):
  - Scoring = PR-AUC ("average_precision"), not accuracy: 82% of people are not KSI.
  - 5 folds grouped by accident (M2's get_cv), not cv=2 and not a random split.
  - Preprocessing (M2's build_preprocessor) is INSIDE the Pipeline, so it is
    refitted on each training fold: no leakage.
  - Test years 2024-2025 are never touched here.
  - Grids include simple models (shallow trees, strong regularisation) to see
    the bias-variance trade-off, not only complex ones.

Why a subsample for step 1: 619,552 training people x 5 folds x many settings is
too slow on a laptop (one logistic regression CV alone takes about 4 minutes;
kNN and SVM grow much faster than linear with the number of rows). We sample
WHOLE accidents, so the grouping is kept. kNN and SVM use a smaller sample
because their cost grows fastest. Step 2 then retrains the best setting of the
models that scale (logistic regression, decision tree, random forest) on all
training years.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import GROUP_COL, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR
from src.evaluation.metrics import best_f2_threshold, evaluate
from src.features.build_features import add_features
from src.features.preprocess import build_preprocessor
from src.features.split import get_cv, get_splits

N_JOBS = 4  # 8 CPUs but limited RAM: each parallel job copies the data

# name -> (preprocessor kind, estimator, grid, subsample size in people)
SEARCH_SPACE = {
    "Logistic regression": (
        "linear",
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10]},
        100_000,
    ),
    # Trees receive a sparse one-hot matrix (about 25 s per deep tree on 80k people),
    # so the tree grids use 30,000 people and a smaller grid (first run did not finish).
    "Decision tree": (
        "tree",
        DecisionTreeClassifier(criterion="gini", class_weight="balanced", random_state=RANDOM_STATE),
        {"model__max_depth": [4, 8, 16, None],
         "model__min_samples_leaf": [1, 100]},
        30_000,
    ),
    "Random forest": (
        "tree",
        RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", max_features="sqrt",
                               n_jobs=2, random_state=RANDOM_STATE),
        {"model__max_depth": [10, None],
         "model__min_samples_leaf": [1, 20]},
        30_000,
    ),
    "k-nearest neighbours": (
        "linear",
        KNeighborsClassifier(),
        {"model__n_neighbors": [5, 25, 101],
         "model__weights": ["uniform", "distance"]},
        20_000,
    ),
    # rbf kernel only: a linear SVM is close to logistic regression (already tested)
    # and its libsvm training did not finish in 2 hours in a first run (C=10).
    # max_iter caps the optimiser so one setting can never block the whole search.
    "SVM": (
        "linear",
        SVC(kernel="rbf", gamma="scale", class_weight="balanced", max_iter=50_000,
            random_state=RANDOM_STATE),
        {"model__C": [0.1, 1]},
        10_000,
    ),
}

SEARCH_CSV = RESULTS_DIR / "automl_v01_search.csv"
BEST_JSON = RESULTS_DIR / "automl_v01_best_params.json"

# Models retrained on all training years in step 2 (the others do not scale).
FULL_DATA_MODELS = ["Logistic regression", "Decision tree", "Random forest"]


def load_training():
    persons = pd.read_parquet(PROCESSED_DIR / "persons.parquet")
    X, y, groups, _, _ = get_splits(add_features(persons))  # test part ignored on purpose
    return X, y, groups


def accident_subsample(X, y, groups, n_people: int):
    """Random sample of whole accidents, about n_people people, reproducible."""
    acc = pd.Series(groups.unique()).sample(frac=1, random_state=RANDOM_STATE)
    sizes = groups.value_counts().loc[acc]
    keep = set(sizes.index[sizes.cumsum() <= n_people])
    mask = groups.isin(keep)
    return X[mask], y[mask], groups[mask]


def make_pipeline(kind: str, estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(kind)), ("model", estimator)])


def _save(all_rows, best) -> None:
    """Write results after EACH model, so a slow model never loses the others."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).round(4).to_csv(SEARCH_CSV, index=False)
    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=1, default=float)


def run_search() -> None:
    X, y, groups = load_training()
    all_rows, best = [], {}
    if BEST_JSON.exists() and "--fresh" not in sys.argv:  # resume: keep models already searched
        best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
        all_rows = pd.read_csv(SEARCH_CSV).to_dict("records")
    for name, (kind, estimator, grid, n_people) in SEARCH_SPACE.items():
        if name in best:
            print(f"skip {name} (already in {BEST_JSON.name}; use --fresh to redo)", flush=True)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] start {name}", flush=True)
        Xs, ys, gs = accident_subsample(X, y, groups, n_people)
        folds = list(get_cv().split(Xs, ys, gs))
        search = GridSearchCV(make_pipeline(kind, estimator), grid, scoring="average_precision",
                              cv=folds, n_jobs=N_JOBS, refit=False, return_train_score=True)
        t = time.time()
        search.fit(Xs, ys)
        secs = time.time() - t
        res = pd.DataFrame(search.cv_results_)
        for _, r in res.iterrows():
            all_rows.append({"model": name, "params": json.dumps({k.replace("model__", ""): v for k, v in r.params.items()}, default=str),
                             "cv_pr_auc_mean": r.mean_test_score, "cv_pr_auc_std": r.std_test_score,
                             "train_pr_auc_mean": r.mean_train_score, "fit_time_s": r.mean_fit_time,
                             "subsample_people": len(Xs)})
        i = res.mean_test_score.idxmax()
        best[name] = {"params": res.params[i], "cv_pr_auc_mean": res.mean_test_score[i],
                      "cv_pr_auc_std": res.std_test_score[i], "train_pr_auc_mean": res.mean_train_score[i],
                      "subsample_people": len(Xs), "search_time_s": round(secs)}
        print(f"{name:22} best PR-AUC {res.mean_test_score[i]:.4f} +/- {res.std_test_score[i]:.4f} "
              f"(train {res.mean_train_score[i]:.4f}) {res.params[i]}  [{len(Xs):,} people, {secs:.0f} s]", flush=True)
        _save(all_rows, best)


def run_full() -> None:
    X, y, groups = load_training()
    with open(BEST_JSON, encoding="utf-8") as f:
        best = json.load(f)
    folds = list(get_cv().split(X, y, groups))
    rows = []
    for name in FULL_DATA_MODELS:
        kind, estimator, _, _ = SEARCH_SPACE[name]
        pipe = make_pipeline(kind, estimator).set_params(**best[name]["params"])
        t = time.time()
        oof = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba", n_jobs=2)[:, 1]
        thr = best_f2_threshold(y, oof)
        for label, threshold in [("0.5", 0.5), ("best F2", thr)]:
            rows.append({"model": name, "threshold_rule": label, **evaluate(y, oof, threshold=threshold),
                         "time_s": round(time.time() - t)})
        np.save(RESULTS_DIR / f"oof_{name.lower().replace(' ', '_')}.npy", oof.astype(np.float32))
        print(f"{name:22} out-of-fold PR-AUC {rows[-1]['pr_auc']:.4f}  ({time.time() - t:.0f} s)", flush=True)
    pd.DataFrame(rows).round(4).to_csv(RESULTS_DIR / "automl_v01_full_train.csv", index=False)


if __name__ == "__main__":
    {"search": run_search, "full": run_full}[sys.argv[1]]()
