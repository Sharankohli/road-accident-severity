# AI AUDIT

Real mistakes made by AI tools that we caught (target: at least 4, at least 1 per member).

## Entry template

### A<n>: <short title>
- **Member:** 
- **Date:** YYYY-MM-DD
- **What the AI produced:** 
- **How we detected it:** 
- **Root cause:** 
- **Fix:** 
- **Proof the fix works:** (numbers, test, plot)

---

### A1: "0 always means missing" would have erased real information
- **Member:** Sharan Raj JANAKIRAMAN (M1)
- **Date:** 2026-11-25
- **What the AI produced:** the AI's first data notes (`docs/DATA_FACTS.md`, finding 3) listed `0` as one of the codes meaning "missing", following the general sentence in the official dictionary. Applied to every column, this would have turned real answers into NaN.
- **How we detected it:** before writing `src/data/clean.py`, we read the dictionary entry of each column and counted the raw codes. In `obs` 0 means "no fixed obstacle" (557,062 vehicles), in `secu1` 0 means "no safety equipment" (77,688 people); `choc`, `obsm`, `infra`, `situ`, `vosp`, `locp` are similar.
- **Root cause:** a general rule from the documentation was trusted without checking each column's own code list.
- **Fix:** per-column rules in `clean.py` (`ZERO_IS_REAL`, `ZERO_IS_MISSING`).
- **Proof the fix works:** after cleaning, `obs == "0"` = 557,062 and `secu1 == "0"` = 77,688 (unchanged); NaN counts equal the raw code counts, e.g. `catv` 1,823 = 1,804 zeros + 19 × "-1", `manv` 42,312 = 42,093 + 219.

### A2: Month and department written in two formats across years
- **Member:** Sharan Raj JANAKIRAMAN (M1)
- **Date:** 2026-11-25
- **What the AI produced:** the first version of `src/data/clean.py` listed `mois`, `jour`, `dep`, `com` as "untouched" columns. But the files write them as `1` in 2019–2021 and `01` from 2022, so January became two different categories.
- **How we detected it:** the drift check in `notebooks/01_data_audit_eda.ipynb` (section 7) showed a huge train/test difference for `mois` (total variation distance 0.29), which is impossible for months.
- **Root cause:** the AI assumed these columns had a stable format and did not check the values per year.
- **Fix:** removed them from `UNTOUCHED`, so they get the same leading-zero normalisation as the other code columns.
- **Proof the fix works:** after the fix `mois` has 12 distinct values and its drift fell from 0.29 to 0.03; `dep` values appear identically in train and test.

### A3: Vehicle codes compared with the wrong format
- **Member:** Sharan Raj JANAKIRAMAN (M1)
- **Date:** 2026-11-20
- **What the AI produced:** an exploratory analysis script mapped vehicle categories using the dictionary's format (`"07"` = car), but the data writes `"7"`.
- **How we detected it:** the group "Other" was implausibly the largest road-user group (about 166,000 people in 2024–2025).
- **Root cause:** the documentation's formatting was trusted without looking at the actual values.
- **Fix:** strip leading zeros before comparing codes (kept in `clean.py` for every code column).
- **Proof the fix works:** 144,576 car occupants (2024–2025) moved from "Other" to "Car"; "Other / unknown" became a small group, and cars now show a KSI rate of 12.5% (see the EDA notebook, section 4).

### A4: Downloads that failed silently
- **Member:** Sharan Raj JANAKIRAMAN (M1)
- **Date:** 2026-11-20
- **What the AI produced:** a download script whose URL list contained Windows line endings (`\r`), so all 28 downloads failed at first; on the retry, `usagers-2022.csv` arrived truncated (about 10 MB of 13 MB) without any error.
- **How we detected it:** missing files, then a file size smaller than the size published on data.gouv.fr.
- **Root cause:** no check that downloaded files were complete.
- **Fix:** `scripts/download_data.py` stores the verified size of every file and refuses a file of the wrong size.
- **Proof the fix works:** the script reports "ok" for all 28 files; `usagers-2022.csv` = 13,013,255 bytes; the loader's row counts match the verified figures (e.g. 871,534 people).

### A5: Logistic smoke-test iteration limit was too low
- **Member:** Xinyu ZHU (M2)
- **Date:** 2026-11-25
- **What the AI produced:** the first local logistic-regression smoke command used `max_iter=50`. The estimator's `.fit()` returned, but scikit-learn emitted a `ConvergenceWarning`; treating return without inspecting warnings as success would have overstated the check.
- **How we detected it:** the actual `.venv/bin/python` command output contained `lbfgs failed to converge after 50 iteration(s)`.
- **Root cause:** the smoke-test configuration imposed an arbitrary low iteration limit and did not initially promote convergence warnings to failures.
- **Fix:** reran with `solver="liblinear"`, `max_iter=1000`, `RANDOM_STATE` from `src/config.py`, and `ConvergenceWarning` promoted to an error. The notebook uses the same guarded configuration.
- **Proof the fix works:** the sample training pipeline completed without a convergence warning in 8 iterations (1,503 training rows, 170 transformed features). This is only an interface smoke test; it reports no model score. Entry left for M2 review.

### A6: Safety-equipment flag initially treated every nonzero code as equipment used
- **Member:** Xinyu ZHU (M2)
- **Date:** 2026-11-25
- **What the AI produced:** the first feature implementation mapped any nonzero value in `secu1`/`secu2` to `equipment_present`. The official ONISR dictionary defines equipment codes 0–9, including 8 for undeterminable equipment. [ONISR data dictionary, p. 12](https://www.onisr.securite-routiere.gouv.fr/sites/default/files/2022-11/Description%20des%20bases%20de%20donn%C3%A9es%20annuelles_2021_1.pdf#page=12)
- **How we detected it:** a final schema/semantics review compared the feature rule against the documented meanings in `M2_GUIDE.md` and the limited descriptions in `DATA_DICTIONARY.md`.
- **Root cause:** code presence was incorrectly equated with known equipment use without a complete code mapping.
- **Fix:** `belt_or_helmet_recorded` is positive when either field contains 1 (belt) or 2 (helmet). The no-equipment flag requires `secu1=0` with `secu2=0` or missing. Both flags remain missing for other combinations. Raw `secu1`/`secu2` remain categorical inputs.
- **Proof the fix works:** focused assertions used codes 0, 1, 2, and 8 (undeterminable), plus missing values. The no-equipment case and belt-or-helmet cases produced their respective flags, and the code 8 and missing-only cases remained NaN. Both preprocessing variants then fit and transformed all 495 held-out sample rows (170 transformed columns). Entry left for M2 review.

### A7: CatBoost's categorical-feature list broke sklearn cloning
- **Member:** SONG Haoyu (M4)
- **Date:** 2026-11-27
- **What the AI produced:** the first `catboost_pipeline()` passed `cat_features=CATEGORICAL_FEATURES`, a Python list, to CatBoost. The pipeline looked compatible with sklearn, but `clone()` raised `RuntimeError: ... constructor either does not set or modifies parameter cat_features` before cross-validation could fit a model.
- **How we detected it:** executing `05_catboost_interpretation.ipynb` reached the real grouped-CV fit and failed. A minimal check reproduced the error with a list and succeeded with a tuple.
- **Root cause:** this CatBoost/sklearn combination copies the mutable parameter container in a way that fails sklearn's constructor-identity check. Checking only frame conversion was not enough to verify estimator compatibility.
- **Fix:** pass `tuple(CATEGORICAL_FEATURES)` and add `test_catboost_pipeline_clones_and_fits_real_estimator` in `tests/test_m4.py`.
- **Proof the fix works:** the regression test clones the actual pipeline, fits CatBoost on a small mixed-type table with missing categories, and verifies two-class probabilities. This is an interface test, not a model-performance claim.

### A8: The default Python wrote a Parquet file the pinned environment could not read
- **Member:** SONG Haoyu (M4)
- **Date:** 2026-11-27
- **What the AI produced:** the data-generation command initially used the system `python`, while the notebooks used the project's pinned Python 3.12 environment. Reading the resulting file with PyArrow 19.0.0 failed with `OSError: Repetition level histogram size mismatch`.
- **How we detected it:** the first cell that loaded the real dataset in `04_ensembles.ipynb` failed, before any model training.
- **Root cause:** the AI did not use the same dependency environment to write and read the processed dataset.
- **Fix:** regenerate it with `.venv/bin/python -m src.data.make_dataset` and use that environment for notebook execution.
- **Proof the fix works:** the notebooks subsequently loaded 619,552 training people from the rebuilt 871,115-person dataset. The raw files and the project split were not changed.

### A9: A tree-obstacle EDA check used the wrong code
- **Member:** SONG Haoyu (M4)
- **Date:** 2026-11-27
- **What the AI produced:** the initial CatBoost notebook labelled `obs=1` as a tree obstacle in a training-EDA cross-check.
- **How we detected it:** a review against `src/data/DATA_DICTIONARY.md` and the executed M1 notebook showed that trees use code `2`; M1's tree rate was 56.3% for 11,123 training people.
- **Root cause:** the AI recalled a category code instead of checking the project's documented mapping.
- **Fix:** use `obs=2` and recompute the table from the training rows. The raw obstacle feature used by the models was already categorical and was not remapped, so this affected only the explanatory EDA check.
- **Proof the fix works:** `results/m4_eda_checks.csv` and the executed CatBoost notebook show the corrected tree-obstacle count and rate, which can be compared directly with M1.
