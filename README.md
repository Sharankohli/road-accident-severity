# Road-Accident Severity Prediction

ESILV A4 Machine Learning project (DIA - 1), 2026–2027.

We predict whether a person involved in a French injury road accident is **killed or seriously injured (KSI)**, and explain which circumstances raise that risk, to help road-safety authorities decide where to act first.

- **Data:** ONISR road-accident open data, 2019–2025 (data.gouv.fr, Licence Ouverte 2.0). 382,721 accidents, 871,115 people, 18.1% KSI.
- **Task:** binary classification on an imbalanced target, plus model interpretation.
- **Split:** train/validate on 2019–2023, test once on 2024–2025, GroupKFold by accident.

## Team

| Member | Module |
|---|---|
| Sharan Raj JANAKIRAMAN | M1 Data and EDA · M3 Baseline and AutoML v0.1 |
| Xinyu ZHU | M2 Preprocessing and features |
| SONG Haoyu | M4 Advanced models and interpretation |
| FOTSING TANKOU DAN KATEL | M4 Advanced models and interpretation |

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

The final notebook records `results/m4_final_plan.json` before producing test predictions. Reopening it reuses the cached predictions. A changed plan is rejected; do not delete the plan to tune against the test years. To reproduce this frozen result set on a fresh checkout, use its result-producing commit and the pinned environment. Later changes under `src/models/` or `src/features/` change the code fingerprint. Reproduction is not a new model-selection opportunity.

Run the checks with `python -m pytest -q`. The M4 checks cover literal-versus-weighted accident bootstrap, paired differences, input alignment, nested stacking's accident isolation, CatBoost cloning/fitting and the frozen-test guard.
