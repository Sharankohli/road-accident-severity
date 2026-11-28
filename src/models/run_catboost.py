# Owner: FOTSING TANKOU DAN KATEL (M4b)
"""CatBoost on the TRAINING years (2019-2023) only: tuning, out-of-fold scores, interpretation.

Run from the repo root:   python -m src.models.run_catboost [tune|oof|interpret|all]

Steps (the test years 2024-2025 are never used here):
  1. tune      grouped 5-fold CV on a subsample of ~100,000 people (whole accidents)
               -> results/catboost_tuning.csv, results/catboost_best_params.json
  2. oof       best setting, one model per shared grouped fold on all training years
               -> results/catboost_train.csv (same columns as results/baseline_train.csv)
                  data/processed/catboost_oof.parquet (person-level, not on GitHub)
               Threshold for precision/recall/F2 = best_f2_threshold on these
               out-of-fold predictions; it will be applied unchanged to the test years.
  3. interpret permutation importance and SHAP, both computed by the fold-1 model on
               accidents it never saw (a subsample of fold-1 validation accidents),
               and a ranked table of high-severity situations from out-of-fold predictions.
               -> results/catboost_permutation_importance.csv, results/catboost_shap_importance.csv,
                  results/catboost_high_risk_situations.csv, report/figures/m4_catboost_*.png
"""
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.inspection import permutation_importance

from src.config import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR
from src.data.eda import wilson_ci
from src.evaluation.metrics import best_f2_threshold, evaluate
from src.features.build_features import add_features
from src.features.preprocess import CATEGORICAL_FEATURES
from src.features.split import get_cv, get_splits
from src.models.automl import accident_subsample  # M3: whole accidents, RANDOM_STATE
from src.models.catboost_model import FEATURES, CatBoostKSI, out_of_fold, to_catboost_frame, tune

TUNE_PEOPLE = 100_000
INTERPRET_PEOPLE = 30_000   # validation people used for permutation importance and SHAP

# Guide values: depth (4, 6, 8), learning_rate (0.03, 0.1), iterations (500-1000), l2_leaf_reg.
# Iterations 500 and 1000 are both scored from each 1000-tree model (see tune()).
# Two stages, because one fit on 80,000 people takes about 1.5-2.5 minutes (depth 4-8, 1000 trees,
# 8 cores; the full search took 4 hours):
#   stage 1: depth x learning_rate with CatBoost's default l2_leaf_reg = 3,
#   stage 2: l2_leaf_reg 1 and 10 around the best stage-1 setting.
GRID_STAGE1 = [{"depth": d, "learning_rate": lr, "l2_leaf_reg": 3.0} for d in (4, 6, 8) for lr in (0.03, 0.1)]
L2_STAGE2 = (1.0, 10.0)
BEST_PARAMS = RESULTS_DIR / "catboost_best_params.json"
OOF_FILE = PROCESSED_DIR / "catboost_oof.parquet"
FOLD1_MODEL = PROCESSED_DIR / "catboost_fold1.cbm"   # reused by the interpretation step
TUNE_PROGRESS = PROCESSED_DIR / "catboost_tuning_progress.csv"  # lets a crashed tuning restart


def load():
    persons = pd.read_parquet(PROCESSED_DIR / "persons.parquet")
    X, y, groups, _, _ = get_splits(add_features(persons))  # test part ignored on purpose
    t = time.time()
    folds = list(get_cv().split(X, y, groups))  # computed once, reused by every step
    print(f"Training years: {len(X):,} people, KSI rate {y.mean():.1%} (folds: {time.time() - t:.0f} s)")
    return X, y, groups, folds


def step_tune(X, y, groups):
    Xs, ys, gs = accident_subsample(X, y, groups, TUNE_PEOPLE)
    print(f"Tuning on {len(Xs):,} people from {gs.nunique():,} accidents")
    folds = list(get_cv().split(Xs, ys, gs))
    t = time.time()
    stage1 = tune(Xs, ys, folds, GRID_STAGE1, progress_file=TUNE_PROGRESS).assign(stage=1)
    top = stage1.iloc[0]
    grid2 = [{"depth": int(top["depth"]), "learning_rate": float(top["learning_rate"]), "l2_leaf_reg": l2}
             for l2 in L2_STAGE2]
    table = pd.concat([stage1, tune(Xs, ys, folds, grid2, progress_file=TUNE_PROGRESS).assign(stage=2)])
    table = table.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    best = table.iloc[0][["depth", "learning_rate", "l2_leaf_reg", "iterations"]].to_dict()
    best = {"depth": int(best["depth"]), "learning_rate": float(best["learning_rate"]),
            "l2_leaf_reg": float(best["l2_leaf_reg"]), "iterations": int(best["iterations"])}

    # Imbalance: compare the best setting with and without class weights (same folds).
    no_weights = tune(Xs, ys, folds, [{k: v for k, v in best.items() if k != "iterations"}
                                      | {"auto_class_weights": None}],
                      check_iterations=(best["iterations"],), progress_file=TUNE_PROGRESS)
    table["auto_class_weights"] = "Balanced"
    no_weights = no_weights.assign(stage=3, auto_class_weights="None")
    table = pd.concat([table, no_weights], ignore_index=True)
    if no_weights["pr_auc_mean"].iloc[0] > table["pr_auc_mean"].iloc[0]:
        best["auto_class_weights"] = None
    print(f"Tuning: {time.time() - t:.0f} s, best {best}")

    table.round(4).to_csv(RESULTS_DIR / "catboost_tuning.csv", index=False)
    BEST_PARAMS.write_text(json.dumps(best, indent=2))


def step_oof(X, y, groups, folds):
    params = json.loads(BEST_PARAMS.read_text())
    t = time.time()
    oof, models = out_of_fold(X, y, folds, params, keep_models=True)
    models[0].model_.save_model(str(FOLD1_MODEL))
    print(f"CatBoost out-of-fold: {time.time() - t:.0f} s")
    thr = best_f2_threshold(y, oof)
    scored_on = "out-of-fold (5 grouped folds)"
    rows = [{"model": "CatBoost", "scored_on": scored_on, **evaluate(y, oof)},
            {"model": "CatBoost (F2 threshold)", "scored_on": scored_on, **evaluate(y, oof, threshold=thr)}]
    results = pd.DataFrame(rows).set_index("model").round(4)
    results.to_csv(RESULTS_DIR / "catboost_train.csv")
    print(results.drop(columns="scored_on").to_string())
    fold_id = np.zeros(len(y), dtype=np.int8)
    for k, (_, valid) in enumerate(folds, start=1):
        fold_id[valid] = k
    pd.DataFrame({"Num_Acc": groups.to_numpy(), "KSI": y.to_numpy(), "fold": fold_id, "proba": oof},
                 index=X.index).to_parquet(OOF_FILE)


def speed_band(vma: pd.Series) -> pd.Series:
    bands = pd.cut(vma, bins=[0, 30, 50, 70, 80, 110, 130],
                   labels=["<=30", "31-50", "51-70", "71-80", "81-110", "111-130"])
    return bands.astype("string").fillna("unknown")


def obstacle(obs: pd.Series) -> pd.Series:
    """Fixed obstacle hit, simplified for the table (codes from the ONISR dictionary)."""
    out = pd.Series("other fixed obstacle", index=obs.index)
    out[obs.eq("0")] = "none"
    out[obs.eq("2")] = "tree"
    out[obs.isna()] = "unknown"
    return out


def high_risk_table(X, oof, threshold, min_people=300):
    """Road user x speed limit x fixed obstacle, ranked by predicted KSI rate."""
    df = pd.DataFrame({
        "road_user": X["road_user"].astype("string").fillna("unknown"),
        "speed_limit": speed_band(X["vma"]),
        "fixed_obstacle": obstacle(X["obs"]),
        "KSI": oof["KSI"], "proba": oof["proba"],
    })
    g = df.groupby(["road_user", "speed_limit", "fixed_obstacle"]).agg(
        people=("KSI", "size"), ksi=("KSI", "sum"), predicted_ksi_rate=("proba", "mean"),
        flagged_share=("proba", lambda p: (p >= threshold).mean()))
    g = g[g.people >= min_people].copy()
    g["observed_ksi_rate"] = g.ksi / g.people
    g["observed_ci_low"], g["observed_ci_high"] = wilson_ci(g.ksi.to_numpy(), g.people.to_numpy())
    cols = ["people", "observed_ksi_rate", "observed_ci_low", "observed_ci_high",
            "predicted_ksi_rate", "flagged_share"]
    return g[cols].sort_values("predicted_ksi_rate", ascending=False).reset_index()


def step_interpret(X, y, groups, folds):
    params = json.loads(BEST_PARAMS.read_text())
    oof = pd.read_parquet(OOF_FILE).loc[X.index]
    train, valid = folds[0]
    model = CatBoostKSI(**params)  # fold-1 model saved by step_oof (trained on folds 2-5)
    model.model_ = CatBoostClassifier().load_model(str(FOLD1_MODEL))
    model.classes_ = model.model_.classes_
    Xv, yv, gv = accident_subsample(X.iloc[valid], y.iloc[valid], groups.iloc[valid], INTERPRET_PEOPLE)
    Xv = Xv[FEATURES]
    print(f"Interpretation on {len(Xv):,} unseen people from {gv.nunique():,} accidents")

    # Permutation importance: PR-AUC lost when one column is shuffled (rows of the column only).
    t = time.time()
    perm = permutation_importance(model, Xv, yv, scoring="average_precision", n_repeats=5,
                                  random_state=RANDOM_STATE, n_jobs=1)
    perm_table = pd.DataFrame({"feature": FEATURES, "pr_auc_drop_mean": perm.importances_mean,
                               "pr_auc_drop_std": perm.importances_std})
    perm_table = perm_table.sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True)
    perm_table.round(5).to_csv(RESULTS_DIR / "catboost_permutation_importance.csv", index=False)
    print(f"Permutation importance: {time.time() - t:.0f} s")

    # SHAP values (CatBoost computes exact tree SHAP itself; shap is used for the plot).
    Xc = to_catboost_frame(Xv)
    sv = model.model_.get_feature_importance(Pool(Xc, cat_features=CATEGORICAL_FEATURES), type="ShapValues")
    sv = sv[:, :-1]  # last column = expected value (base log-odds)
    shap_table = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    shap_table = shap_table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_table.round(5).to_csv(RESULTS_DIR / "catboost_shap_importance.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = perm_table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.feature, top.pr_auc_drop_mean, xerr=top.pr_auc_drop_std, color="#4C72B0")
    ax.set_xlabel("Drop in PR-AUC when the column is shuffled (mean of 5 repeats)")
    ax.set_title("CatBoost permutation importance (top 15, unseen fold-1 accidents)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "m4_catboost_permutation_importance.png", dpi=150)
    plt.close(fig)

    # Beeswarm: colour = value of numeric columns; category columns have no order, so grey.
    order = shap_table.feature.head(15).tolist()
    pos = [FEATURES.index(f) for f in order]
    colour = Xc[order].apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name not in CATEGORICAL_FEATURES
                             else pd.Series(np.nan, index=c.index))
    plt.figure()
    shap.summary_plot(sv[:, pos], colour.to_numpy(dtype=float), feature_names=order, max_display=15,
                      show=False, plot_size=(8, 6))
    plt.title("CatBoost SHAP values (top 15; colour only for numeric columns)")
    plt.xlabel("SHAP value (impact on KSI log-odds)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "m4_catboost_shap_summary.png", dpi=150)
    plt.close()

    threshold = pd.read_csv(RESULTS_DIR / "catboost_train.csv", index_col="model").loc[
        "CatBoost (F2 threshold)", "threshold"]
    table = high_risk_table(X, oof, threshold)
    table.round(4).to_csv(RESULTS_DIR / "catboost_high_risk_situations.csv", index=False)
    print(table.head(15).round(3).to_string())


def main(step: str = "all") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups, folds = load()
    if step in ("tune", "all"):
        step_tune(X, y, groups)
    if step in ("oof", "all"):
        step_oof(X, y, groups, folds)
    if step in ("interpret", "all"):
        step_interpret(X, y, groups, folds)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
