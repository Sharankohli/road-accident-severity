# M2 Guide: Preprocessing, Feature Engineering and Feature Selection

**Owner:** Xinyu ZHU · **Branch:** `m2/<task>` (e.g. `m2/split`, `m2/preprocessing`) · **Reviewer:** Sharan (M3 uses your work first)

This guide says **what** M2 must deliver, the **rules**, and the **facts about the data** you need. The **choices** (which encoding, which features, which selection method) are yours: you will explain and defend them at the oral, so read the "Your decisions" parts carefully.

---

## 0. Start here (30 minutes)

```powershell
git checkout main
git pull
python scripts/download_data.py        # 28 raw files, about 211 MB
python -m src.data.make_dataset        # builds data/processed/persons.parquet (about 1 min)
git checkout -b m2/split
```

Then read, in this order:
1. `src/data/DATA_DICTIONARY.md`: every column, its meaning, type and missing %.
2. `notebooks/01_data_audit_eda.ipynb`, **section 8 "Findings and decisions"**: what M1 found that affects you.
3. `src/config.py`: seed, paths, years, target. Always import from here.

For quick tests, use `data/processed/persons_sample.parquet` (1,998 people, whole accidents). Use the full file for real results.

---

## 1. Your place in the pipeline

```
M1 persons.parquet ──► M2 (you) ──► M3 baselines + AutoML (Sharan) ──► M4 advanced models (Song, Fotsing)
```

**Input:** `data/processed/persons.parquet`: 871,115 people, 46 columns, one row per person, target `KSI` (18.1% = 1).
**Output:** 3 Python functions that M3 and M4 import. Everyone uses the **same** split and the **same** preprocessing, so all results are comparable.

---

## 2. What you must deliver

| # | File | Function(s) | Used by |
|---|---|---|---|
| 1 | `src/features/split.py` | `get_splits(df)`, `get_cv()` | M3, M4 |
| 2 | `src/features/build_features.py` | `add_features(df)` | M3, M4 |
| 3 | `src/features/preprocess.py` | `build_preprocessor(kind)` | M3, M4 |
| 4 | `src/features/select.py` (optional but recommended) | feature-selection step usable inside a `Pipeline` | M3 |
| 5 | `notebooks/02_preprocessing_features.ipynb` | explains and tests everything above, incl. a **leakage demo** | report, oral |
| 6 | `PROMPT_LOG.md` + `AI_AUDIT.md` | at least 1 entry each, about M2 | grade |

The function **names and outputs** below are a contract: M3 and M4 will code against them. If you need to change one, say so in the group chat first.

---

## 3. Task 1: the split (`split.py`)

### What it must do
```python
def get_splits(df):
    """Return X_train, y_train, groups_train, X_test, y_test."""

def get_cv():
    """Return the cross-validation object used by everyone on the training set."""
```

### Rules (already decided in the spec, do not change)
- **Train** = years 2019–2023 (`TRAIN_YEARS` in `config.py`): 619,552 people, 273,226 accidents.
- **Test** = years 2024–2025 (`TEST_YEARS`): 251,563 people. Used **once**, at the very end (M4 final table). Never for tuning.
- **Cross-validation inside train, grouped by accident** (`GROUP_COL = "Num_Acc"`). 82% of accidents have 2+ people who share the same circumstances; if they are split between a training fold and a validation fold, the model "recognises" the accident and scores look better than reality.
- `X` must **not** contain `KSI`. `groups_train` = the `Num_Acc` of each training row.
- Use `RANDOM_STATE` from `config.py`.

### Hints
- scikit-learn has `GroupKFold` and `StratifiedGroupKFold` (groups + keeps the KSI rate similar in every fold). Read both docs pages and choose; 5 folds is a common default.
- Write a check that **no `Num_Acc` appears in both** a training fold and its validation fold.

---

## 4. Task 2: new features (`build_features.py`)

### The key idea: "stateless" vs "fitted" transformations
- **Stateless** = computed from the row itself, with no statistics from other rows. Example: `age = year - an_nais`. Safe to do **before** the split, in `add_features(df)`.
- **Fitted** = needs statistics learned from data (a mean, a list of categories, a scaler, a target encoding). Must be done **inside the `Pipeline`** (Task 3), so it is learned on training folds only.

If you are unsure whether something is stateless, it is probably fitted: put it in the Pipeline.

### Candidate features (you choose, justify, and test them)
| Idea | From | Why it may help |
|---|---|---|
| `age` | `year - an_nais` | Age 75+ has 33% KSI vs 15% at 25–44 |
| `hour`, `night` | `hrmn` (`"HH:MM"`, no bad values) | Night without lights: 29% KSI |
| `weekday`, `weekend` | `year`, `mois`, `jour` | Traffic patterns |
| `belt_used`, `helmet_used`, `no_equipment` | `secu1`, `secu2`, `secu3` **together** (any of the 3) | Strongest association with KSI (Cramér's V 0.29). Codes: 0 none, 1 belt, 2 helmet (check the dictionary for the rest) |
| `road_user` group | `catu`, `catv` | Readable groups; M1 helper `src/data/eda.py → road_user()` |
| `n_vehicles`, `n_people` in the accident | count rows by `Num_Acc` / `id_vehicule` | Crash size |
| `other_vehicle` category (e.g. hit by a heavy truck) | the other vehicles of the same `Num_Acc` | Being hit by a truck is not the same as by a bike |
| `is_overseas` | `dep` starts with `97` | Different road networks |

⚠️ **Never** build a feature from another person's outcome (e.g. "number of people killed in this accident"). That is target leakage: it uses the answer.

---

## 5. Task 3: the preprocessing pipeline (`preprocess.py`)

### What it must do
```python
def build_preprocessor(kind="linear"):
    """Return an UNFITTED sklearn ColumnTransformer.
    kind="linear": for logistic regression, SVM, kNN (needs scaling, one-hot).
    kind="tree":   for decision trees, random forest, boosting (no scaling needed)."""
```
**Unfitted** means M3/M4 put it inside their own `Pipeline` and call `.fit()` on training folds. Never return something already fitted on all the data.

### Facts from M1 you must handle
| Fact | Where it comes from | What to think about |
|---|---|---|
| Codes are **text** (`"7"`, not `7`); numbers are floats | `DATA_DICTIONARY.md` | Codes are categories → encode them, never scale them as numbers |
| `0` is a **real category** in `obs`, `obsm`, `choc`, `infra`, `situ`, `vosp`, `secu1-3`, `locp` | M1 cleaning | Don't treat 0 as missing |
| `actp`, `etatp`, `locp` are ~90% empty because they **only apply to pedestrians**; `secu2`, `secu3` empty when there is no 2nd/3rd item | EDA section 2 | "Missing" has a meaning here (missing not at random). A `"missing"` category is often better than filling with the most frequent value |
| `com` has **25,588** distinct values, `dep` 107 | dictionary | One-hot on `com` = 25,588 columns. Drop it, group it, or use target encoding (inside the Pipeline only) |
| `road_user` and `catv` are nearly identical (Cramér's V 0.91) | EDA section 6 | Keep only one |
| `secu3` is filled for 1.3% of people in train vs 9.5% in test (recording changed) | EDA section 7 | Raw `secu3` may hurt on 2024–2025; combined equipment flags are safer |
| `larrout` is 90% empty | EDA section 2 | Keep with a "missing" indicator, or drop: your call |
| IDs (`person_id`, `Num_Acc`, `id_vehicule`), `year`, `lat`/`long`, `jour` | — | Decide: IDs must **never** be features. `year` as a feature is risky with a time split |

### Your decisions (write the reason for each in the notebook)
1. Which columns go in, which are dropped.
2. Missing values: `"missing"` category, most frequent value, median, or an indicator column?
3. Categorical encoding: one-hot (with `min_frequency` / `handle_unknown` for rare and new categories), ordinal for trees, or target encoding for high-cardinality columns?
4. Scaling numeric columns for the `"linear"` kind.
5. Class imbalance (18% KSI): `class_weight="balanced"` in the model, or resampling (SMOTE)? If SMOTE: it must be inside an `imblearn.pipeline.Pipeline`, never applied to the whole training set before CV.

---

## 6. Task 4: feature selection (`select.py`)

LAB4 Part I taught filter, wrapper and embedded methods, and the trap **"selecting features before cross-validation"**. That trap is exactly what you must avoid:

- ❌ Wrong: compute mutual information / chi² on all training data, keep the top 20, **then** run CV → the validation folds already influenced the choice → scores too optimistic.
- ✅ Right: put the selector **inside the Pipeline** (e.g. `SelectKBest`, `SelectFromModel`, L1 logistic regression), so it is refitted on each training fold.

Deliver a selection step M3 can plug in, and in the notebook compare: all features vs selected features (same CV, same metric).

---

## 7. Task 5: the notebook `02_preprocessing_features.ipynb`

First cell: `# Owner: Xinyu ZHU, M2`. Sections:
1. The split: sizes, KSI rate per fold, proof that no accident is in two folds.
2. New features: what, why, a quick check (e.g. KSI rate by `night`).
3. The preprocessor: columns in/out, shape after transformation for `"linear"` and `"tree"`.
4. **Leakage demo (very valuable for the grade):** run the same simple model twice:
   - wrong: fit the encoder/scaler (or SMOTE, or feature selection) on all training data, then CV;
   - right: everything inside the Pipeline.
   Show the score difference in a small table and explain it. Also compare `KFold` (people split randomly) vs grouped CV.
5. Feature selection comparison.
6. Short "decisions" table (like M1's section 8).

Figures: title, labelled axes, saved in `report/figures/` as `m2_*.png`.

---

## 8. Definition of done (check before opening your PR)

- [ ] `from src.features.split import get_splits, get_cv` works; train = 619,552 rows, test = 251,563 rows.
- [ ] No `Num_Acc` in both the training and validation part of any fold (check in the notebook).
- [ ] `KSI` and ID columns are not in `X`.
- [ ] `build_preprocessor("linear")` and `("tree")` return **unfitted** objects; `Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression())]).fit(X_train, y_train)` runs.
- [ ] The preprocessor works on the **test** years without error (new or rare categories handled).
- [ ] Every fitted step (imputer, encoder, scaler, SMOTE, selector) is inside the Pipeline.
- [ ] `RANDOM_STATE` and paths come from `src/config.py`.
- [ ] Notebook runs top to bottom; leakage demo included.
- [ ] Any new library added to `requirements.txt` with its exact version.
- [ ] 1 entry in `PROMPT_LOG.md` and 1 real AI mistake in `AI_AUDIT.md`, written by you.
- [ ] PR opened with the template; Sharan as reviewer.

---

## 9. Using AI for M2 (allowed, but logged)

Good uses: explaining a scikit-learn class, drafting code you then test, checking your reasoning. Before trusting AI code, check these classic M2 mistakes (each one, if you catch it, is an `AI_AUDIT.md` entry):
- scaler / encoder / imputer `.fit()` on the whole dataset before the split;
- SMOTE applied before cross-validation;
- feature selection done once, outside CV;
- random `train_test_split` instead of the year split, or `KFold` that ignores `Num_Acc`;
- treating code columns (`catv`, `catr`, ...) as numbers and scaling them;
- invented parameters (check every argument in the official scikit-learn docs).

For `PROMPT_LOG.md`: save your first simple prompt, then the improved prompt with constraints (schema, leakage rules, the checks above), and what changed.

---

## 10. Questions?
Ask in the group chat. Data questions → Sharan (M1). If something in this guide is wrong or unclear, open an issue or edit this file in your PR.
