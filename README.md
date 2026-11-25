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

**Before contributing, read [GUIDE.md](GUIDE.md).**

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
