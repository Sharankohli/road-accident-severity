# Team Guide

How we work together on this repo. Read it once before your first change.

## 1. The pipeline and who owns what

```
M1 Data (Sharan) ─► M2 Features (Xinyu) ─► M3 Baseline + AutoML (Sharan) ─► M4 Advanced + Interpretation (Song, Fotsing)
```

| Module | Owner | Folder / files you own |
|---|---|---|
| M1 Data and EDA | Sharan | `src/data/`, `notebooks/01_*` |
| M2 Preprocessing and features | Xinyu | `src/features/`, `notebooks/02_*` |
| M3 Baseline and AutoML v0.1 | Sharan | `src/models/baseline.py`, `src/models/automl.py`, `notebooks/03_*` |
| M4 Advanced models and interpretation | Song, Fotsing | `src/models/advanced/`, `notebooks/04_*`, `notebooks/05_*` |
| Shared | everyone | `src/config.py`, `src/evaluation/` (change only through a PR that everyone sees) |

**Rule:** only edit files you own. If you need a change in someone else's file, ask them or open an issue.

## 2. Handovers between modules (interface contract)

Each module hands over a **file or a function**, always with the same name and format, so everyone can work in parallel.

| From → To | Handover | Status |
|---|---|---|
| M1 → M2 | `data/processed/persons.parquet`: one row = one person, columns listed below | draft |
| M1 → all | `data/processed/persons_sample.parquet`: 2,000 rows, same columns, for early testing | draft |
| M2 → M3, M4 | `src/features/split.py` → `get_splits()` returns train/test DataFrames and CV folds (GroupKFold by `Num_Acc`) | draft |
| M2 → M3, M4 | `src/features/preprocess.py` → `build_preprocessor()` returns an unfitted scikit-learn `ColumnTransformer` | draft |
| M3 → M4 | `src/evaluation/metrics.py` → `evaluate(y_true, y_proba)` returns PR-AUC, recall, precision, F2, ROC-AUC | draft |
| M3 → M4 | `results/baseline.csv`: baseline scores to beat | draft |

**`persons.parquet` columns (draft, M1 finalises):**
- Keys: `Num_Acc`, `id_vehicule`, `id_usager`, `year`
- Target: `KSI` (1 = killed or hospitalised > 24 h, 0 = otherwise). The raw `grav` column is **removed**.
- Features: circumstance variables from the 4 tables (time, light, weather, area, road, speed limit, vehicle, obstacle, user type, age, sex, safety equipment). Missing values are coded as `NaN` (all the raw codes -1, 0, empty, N/A are converted).

A handover changes only through a PR that the receiving person approves.

## 3. Git workflow

**Never push to `main`.** `main` always holds working, reviewed code.

1. Update your local copy:
   ```bash
   git checkout main
   git pull
   ```
2. Create a branch for one task:
   ```bash
   git checkout -b m2/preprocessing-pipeline
   ```
   Branch names: `<module>/<short-task>`, e.g. `m1/data-cleaning`, `m3/baselines`, `m4/catboost`.
3. Work, then commit small steps:
   ```bash
   git add src/features/preprocess.py
   git commit -m "M2: add one-hot encoding for categorical columns"
   ```
   Commit message: `<module>: <what you did>`, in English.
4. Push your branch:
   ```bash
   git push -u origin m2/preprocessing-pipeline
   ```
5. Open a **Pull Request** on GitHub into `main`, and fill in the template.
6. **One teammate who is not the author approves it.** Reviewer by default = the next module in the pipeline (M1 → M2 → M3 → M4 → M1).
7. Sharan merges it. Then delete the branch.

## 4. Code rules

- **Seed:** use `RANDOM_STATE` from `src/config.py`. Never type `42` yourself.
- **Paths:** use the paths in `src/config.py`. No `C:\Users\...`.
- **No data leakage:** every imputer, encoder, scaler, resampler (SMOTE) and feature selector goes **inside a scikit-learn `Pipeline`**, fitted on training folds only. The test set (2024–2025) is used **once**, at the end.
- **Same split and same metrics for everyone:** always use `get_splits()` and `evaluate()`. No private splits.
- **Notebooks:** numbered (`01_data_audit.ipynb`), owner on the first line (`# Owner: Xinyu ZHU, M2`). Reusable code goes into `src/`. Clear outputs before committing if they are large.
- **Figures:** title, labelled axes with units, saved in `report/figures/`.
- **New library:** add it to `requirements.txt` with its exact version (`pip show <name>`), in your PR.
- **Data:** never commit data files. They are in `.gitignore`.
- Everything in English.

## 5. AI use (graded)

We are allowed to use AI, but **every use must be disclosed** (undisclosed use = −4 points).

- `PROMPT_LOG.md`: each member adds **at least 1** important prompt about their module: the first simple prompt, then the improved one with constraints, and what changed.
- `AI_AUDIT.md`: each member records **at least 1 real AI mistake** they caught (e.g. leakage, wrong metric, wrong assumption): what was wrong, how you found it, the fix, and proof the fix works.
- Raw evidence (screenshots, chat exports) goes in `ai_governance/`.
- Write these **when it happens**, not at the end.

## 6. Communication

- Group chat for quick questions.
- One 30-minute meeting per week. Each person says: what I finished, what I do next, what blocks me.
- Blocked for more than a day? Say it in the chat.

## 7. Timeline

| Week | Goal |
|---|---|
| 1 | M1 publishes the data contract and sample; others start on the sample |
| 2 | Clean data, preprocessing pipeline, baselines |
| 3 | AutoML v0.1, ensembles, out-of-course algorithm |
| 4 | Final evaluation with confidence intervals, report, zip |
