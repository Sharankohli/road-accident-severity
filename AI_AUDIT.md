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
- **What the AI produced:** the first feature implementation mapped any nonzero value in `secu1`/`secu2` to `equipment_present`, although the available schema only documents 0 = none, 1 = belt, and 2 = helmet and does not define the remaining codes.
- **How we detected it:** a final schema/semantics review compared the feature rule against the documented meanings in `M2_GUIDE.md` and the limited descriptions in `DATA_DICTIONARY.md`.
- **Root cause:** code presence was incorrectly equated with known equipment use without a complete code mapping.
- **Fix:** the combined positive flag now recognizes only documented codes 1/2; the no-equipment flag requires documented code 0; all other code combinations remain missing. Raw `secu1`/`secu2` remain categorical inputs.
- **Proof the fix works:** focused assertions set `secu1`/`secu2` to documented 0, 1, and 2, unsupported 8, and missing values: 0 produced the no-equipment flag, 1/2 produced the documented positive flag, and 8/missing remained NaN. Both preprocessing variants then fit and transformed all 495 held-out sample rows (170 transformed columns). Entry left for M2 review.
