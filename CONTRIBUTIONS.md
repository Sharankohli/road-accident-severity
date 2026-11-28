# CONTRIBUTIONS

| Member | Module | Main files | Main contributions |
|---|---|---|---|
| Sharan Raj JANAKIRAMAN | M1 Data and EDA; M3 Baseline and AutoML v0.1 | `src/data/`, `src/models/baseline.py`, `src/models/automl.py` | (to fill) |
| Xinyu ZHU | M2 Preprocessing and features | `src/features/` | (to fill) |
| SONG Haoyu | M4a Ensembles and error analysis | `src/evaluation/bootstrap.py`, `src/models/ensembles.py` | (to fill) |
| FOTSING TANKOU DAN KATEL | M4b CatBoost, interpretation and final test table | `src/models/catboost_model.py`, `src/models/run_catboost.py`, `notebooks/05_catboost_interpretation.ipynb` | CatBoost (out-of-course algorithm) with paper summary, tuning on a subsample of whole accidents and out-of-fold scores on the training years (PR-AUC 0.647 vs 0.589 for the logistic baseline); calibration check; permutation importance, SHAP (also by category), comparison with the logistic odds ratios and table of high-severity situations; final evaluation on the test years (`src/models/final_evaluation.py`, branch `m4/final-evaluation`, runs once the M4a bootstrap is merged) |

## M4 split

M4 has two owners, so it is split into two halves, each with its own branches, code and oral explanation:

- **M4a (SONG Haoyu):** bootstrap confidence intervals, ensembles (bagging, boosting, voting, stacking), error analysis. Branches `m4/bootstrap`, `m4/ensembles`.
- **M4b (FOTSING TANKOU DAN KATEL):** CatBoost and its paper, interpretation of risk factors, final evaluation on the test years. Branches `m4/catboost`, `m4/final-evaluation`.
