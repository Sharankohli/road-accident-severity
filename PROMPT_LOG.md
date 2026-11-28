# PROMPT LOG

Important AI interactions (target: 5 entries, at least 1 per member). Each entry shows how a simple prompt was improved with constraints.

## Entry template

### P<n>: <topic>
- **Member:** 
- **Date:** YYYY-MM-DD
- **Tool:** (e.g. Claude)
- **Goal:** 
- **First prompt (simple):** 
- **Problem with the answer:** 
- **Improved prompt (with constraints):** (data schema, leakage rules, assumptions, checks to run)
- **What we kept / changed:** 

---

### P1: Building the person-level dataset (M1 data pipeline)
- **Member:** Sharan Raj JANAKIRAMAN (M1)
- **Date:** 2026-11-25
- **Tool:** Claude (Claude Code in the Claude desktop app), with the project instruction file `CLAUDE.md` and our team rules
- **Goal:** turn the 28 raw ONISR CSV files into one clean table, one row per person, with the KSI target and no leakage.
- **First prompt (simple):** "let's start my M1 part. im gonna use vs code for this"
- **Problem with the answer:** a prompt this short leaves every important choice to the AI: what "missing" means, how to join tables that are not one-to-one, which columns are leakage, which years may be looked at. The risks were real in this dataset: the official dictionary's general rule "0 = not filled in" is wrong for several columns, and the `lieux` table has several rows per accident from 2023, so a naive join copies people.
- **Improved prompt (with constraints):** the work was then driven step by step ("Yes, step 2", "Yes, step 3", ...), and each step was constrained by our written rules:
  - schema: one row per person; keys `person_id`, `Num_Acc`, `id_vehicule`, `year`; target `KSI` = `grav` in {2, 3}; `grav` removed (target leakage);
  - missing values: check the official dictionary **column by column**, never one rule for all columns;
  - joins: must not change the number of people (`validate="many_to_one"` + row-count check);
  - EDA decisions on training years 2019–2023 only; test years only for drift;
  - checks to run after each step: compare row counts and code counts with the figures verified earlier (`docs/DATA_FACTS.md`).
- **What we kept / changed:** kept the loader/cleaner/joiner split in `src/data/`. Changed: per-column rules for 0 (AI_AUDIT A1), one location row per accident (decision D20), month/department format fix (AI_AUDIT A2), impossible speed limits and ages set to missing.

### P2: M2 split, feature engineering, and preprocessing handoff
- **Member:** Xinyu ZHU (M2)
- **Date:** 2026-11-25
- **Tool:** GitHub Copilot in VS Code
- **Goal:** implement the shared year split, accident-grouped CV, row-wise features, fold-fitted preprocessing, and a runnable notebook without evaluating on the final test set.
- **First prompt (simple):** No separate simple prompt was sent. The actual initial prompt in this conversation began, "You are working on the ESILV machine learning group project road-accident-severity. I am M2, responsible for preprocessing and feature engineering," and continued with the full file contracts, data rules, environment constraints, checks, and handoff requirements recorded in this entry.
- **Problem with the answer:** the first logistic-regression smoke command used `max_iter=50` and emitted a convergence warning. Final schema review also caught that the initial equipment flag treated every nonzero code as equipment used, although only codes 0/1/2 were documented.
- **Improved prompt (with constraints):** the same request specified the actual M1 schema and findings, exact 2019-2023 / 2024-2025 split sizes, `Num_Acc`-grouped stratified CV, forbidden IDs/outcomes, no invented code meanings, no test-set tuning/performance, training-fold-only fitted transforms, no SMOTE, and requested a training-only logistic pipeline smoke test. Refinement: configure the smoke test with `RANDOM_STATE`, `liblinear`, `max_iter=1000`, and warnings-as-errors; derive equipment flags only from documented values and leave unsupported codes unknown.
- **What we kept / changed:** kept the row-wise feature boundary and pipeline-contained fitted preprocessing. The revised logistic check converged in 8 iterations on the sample training rows. Equipment flags were narrowed to documented codes; focused 0/1/2/8/missing assertions passed, and both preprocessors transformed held-out sample rows. No model performance was computed.

### P3: Complete M4 in runnable notebooks
- **Member:** SONG Haoyu (M4)
- **Date:** 2026-11-27
- **Tool:** OpenAI Codex
- **Goal:** complete the M4 models, accident-level confidence intervals, error analysis, interpretation and final comparison, using conversational English in the notebooks.
- **First prompt (simple):** "完成m4a" (complete M4a).
- **Improved prompt (with constraints):** "使用notebook，完成m4的所有任务，记住对于notebook中的文字说明使用口语化英语" (use notebooks, complete all M4 tasks, and write their explanations in conversational English). Later, the user requested the tasks in `SONG_ENSEMBLES_TODAY.md`, including the shared-sample comparison and PR delivery. The user explicitly chose to wait for the real M3 AutoML handoff instead of inventing a replacement search; work resumed after PR #8 was merged.
- **Problem with the answer:** the first CatBoost pipeline passed a mutable list as `cat_features`, which failed when scikit-learn cloned the model. The initial dataset was also written with a different PyArrow environment and could not be read by the project's pinned PyArrow.
- **What we kept / changed:** reused M3's saved best settings; kept preprocessing within each fit; used nested grouped CV for stacking; changed CatBoost's feature-name container to a tuple and added a real fit test; rebuilt the processed data with the pinned Python 3.12 environment. The test years are reserved for a frozen final plan after training choices are complete.
