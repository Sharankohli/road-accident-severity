"""Shared settings for the whole project. Import from here, never hard-code.

Example:
    from src.config import RANDOM_STATE, RAW_DIR
"""
from pathlib import Path

# One seed for everything (splits, models, bootstrap) -> reproducible results.
RANDOM_STATE = 42

# Paths are relative to the repo root, so the code works on every laptop.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # downloaded CSVs (never edited)
PROCESSED_DIR = DATA_DIR / "processed"  # outputs of Module 1
RESULTS_DIR = ROOT / "results"          # metric tables (CSV)
FIGURES_DIR = ROOT / "report" / "figures"

YEARS = list(range(2019, 2026))
TRAIN_YEARS = list(range(2019, 2024))   # 2019-2023: training + validation
TEST_YEARS = [2024, 2025]               # final test, used once

TARGET = "KSI"  # 1 = killed or hospitalised > 24 h, 0 = otherwise
GROUP_COL = "Num_Acc"  # people of the same accident stay in the same fold
