# Road-Accident Severity Prediction

ESILV A4 Machine Learning project (DIA - 1), 2026–2027.

We predict whether a person involved in a French injury road accident is **killed or seriously injured (KSI)**, and explain which circumstances raise that risk, to help road-safety authorities decide where to act first.

- **Data:** ONISR road-accident open data, 2019–2025 (data.gouv.fr, Licence Ouverte 2.0). 382,721 accidents, 871,115 people, 18.1% KSI.
- **Task:** binary classification on an imbalanced target, plus model interpretation.
- **Split:** train/validate on 2019–2023, test once on 2024–2025, GroupKFold by accident.

## Team

| Member | Module |
|---|---|
| Sharan Raj JANAKIRAMAN | M1 Data and EDA · M3 Baselines, AutoML v0.1 and v0.2, error analysis |
| Xinyu ZHU | M2 Split, features, preprocessing, feature selection and leakage demo |
| SONG Haoyu | M4a Ensembles, bootstrap intervals, CatBoost and the frozen final test |
| FOTSING TANKOU DAN KATEL | M4b CatBoost independent check, interpretation and recommendations |

## Main result

Final model **CatBoost**, scored once on the unseen test years 2024–2025: **PR-AUC 0.630** (95% accident-bootstrap interval 0.626–0.635), against 0.575 for the logistic baseline (paired gain +0.055, interval 0.052–0.058) and a no-skill level of 0.18. Recall 0.88 at the F2 threshold chosen on training out-of-fold predictions. Details: `results/final_test.csv`, `notebooks/05b_final_evaluation.ipynb`.

## Run order

All notebooks are in `notebooks/` and use only the training years 2019–2023, except `05b` (the one test pass).

| Step | Notebook / command | Owner |
|---|---|---|
| Data | `python scripts/download_data.py`, then `python -m src.data.make_dataset` | M1 |
| 01 | `01_data_audit_eda.ipynb`: data audit and EDA | M1 |
| 02 | `02_preprocessing_features.ipynb`: split, folds, features, preprocessing | M2 |
| 02b | `02b_feature_selection.ipynb`: feature selection and leakage demonstrations | M2 |
| 03 | `03_baselines_automl.ipynb`: baselines, AutoML v0.1, bias-variance, error analysis | M3 |
| 03b | `03b_automl_v02.ipynb`: corrected grid search on equal samples (`python -u -m src.models.automl_v2 search`, `equal`, `full`) | M3 |
| 04 | `04_ensembles.ipynb`: bagging, boosting, voting, nested stacking, bootstrap | M4a |
| 05 | `05_catboost_interpretation.ipynb`: CatBoost, SHAP, permutation importance | M4a |
| 05b | `05b_final_evaluation.ipynb`: frozen choices, one pass on 2024–2025 | M4a |
| 05c | `05c_catboost_independent_check.ipynb`: independent CatBoost and high-risk situations (`python -m src.models.run_catboost_check`) | M4b |

## Quick start

```bash
git clone https://github.com/Sharankohli/road-accident-severity.git
cd road-accident-severity
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_data.py  # downloads the 28 CSV files into data/raw/
```

## Repository structure

```
road-accident-severity/
├── data/               raw/ and processed/ data (not on GitHub)
├── scripts/            download_data.py
├── src/
│   ├── config.py       seed, paths, years, target (shared by everyone)
│   ├── data/           M1: loading, cleaning, joining
│   ├── features/       M2: split, preprocessing pipeline, features
│   ├── models/         M3 baselines and AutoML, M4 advanced models
│   └── evaluation/     shared metrics and bootstrap confidence intervals
├── notebooks/          numbered notebooks (01_, 02_, ...)
├── results/            metric tables (CSV)
├── report/             final report (PDF) and figures/
├── ai_governance/      supporting evidence for AI use
├── PROMPT_LOG.md       key AI prompts (≥1 per member)
├── AI_AUDIT.md         AI mistakes we caught (≥1 per member)
├── CONTRIBUTIONS.md    who did what
└── requirements.txt    pinned dependencies
```

## M4 notebooks

Use Python 3.12 and the pinned requirements. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python -m src.data.make_dataset
python -m ipykernel install --user --name road-accident-m4 --display-name "Python (road-accident M4)"
```

Select **Python (road-accident M4)** in Jupyter or VS Code, then run these notebooks in order:

1. `notebooks/04_ensembles.ipynb`: the common 30,000-person comparison, full training OOF predictions, nested grouped stacking, accident-bootstrap intervals and error analysis.
2. `notebooks/05_catboost_interpretation.ipynb`: paper discussion, CatBoost search and OOF predictions, held-out permutation importance, native CatBoost SHAP and risk-situation tables.
3. `notebooks/05b_final_evaluation.ipynb`: frozen model choices and thresholds, one test prediction pass, paired intervals and the 2024/2025 comparison.

For command-line execution, use `python scripts/execute_notebook.py notebooks/04_ensembles.ipynb` (and the other notebook paths). It saves completed cells if a later step fails. Training is substantial, especially nested stacking. Completed fits are cached under ignored `data/m4/` and reused only when their signatures match. CSV tables go in `results/`; plots go in `report/figures/`.

The final notebook records `results/m4_final_plan.json` before producing test predictions. Reopening it reuses the cached predictions. A changed plan is rejected; do not delete the plan to tune against the test years. To reproduce this frozen result set on a fresh checkout, use its result-producing commit `3093495` and the pinned environment. Later changes under `src/models/` or `src/features/` change the code fingerprint. Reproduction is not a new model-selection opportunity.

Run the checks with `python -m pytest -q`. The M4 checks cover literal-versus-weighted accident bootstrap, paired differences, input alignment, nested stacking's accident isolation, CatBoost cloning/fitting and the frozen-test guard.
