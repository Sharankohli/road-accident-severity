# Owner: Sharan Raj JANAKIRAMAN (M3)
"""Score the baselines that need no preprocessing, on the TRAINING years only.

Run from the repo root:   python -m src.models.run_baselines
Output: results/baseline_train.csv

Why training years only: the test years (2024-2025) are used once, at the very
end, for all models together. These two baselines learn nothing (the rule is
hand-written, the majority model only stores the KSI rate), so scoring them on
the training years does not overfit.
Temporary split: years from config.py (same rule M2's get_splits() will use).
"""
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR, TARGET, TRAIN_YEARS
from src.evaluation.metrics import evaluate
from src.models.baseline import ExpertRuleBaseline, majority_baseline


def main() -> None:
    persons = pd.read_parquet(PROCESSED_DIR / "persons.parquet")
    train = persons[persons["year"].isin(TRAIN_YEARS)]
    X, y = train.drop(columns=TARGET), train[TARGET]

    rows = []
    for name, model in [("Majority class", majority_baseline()), ("Expert rule", ExpertRuleBaseline())]:
        model.fit(X, y)
        rows.append({"model": name, **evaluate(y, model.predict_proba(X)[:, 1])})

    results = pd.DataFrame(rows).set_index("model").round(4)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "baseline_train.csv")
    print(f"Training years {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}: {len(train):,} people, KSI rate {y.mean():.1%}\n")
    print(results.to_string())


if __name__ == "__main__":
    main()
