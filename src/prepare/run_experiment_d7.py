# src/prepare/run_experiment_d7.py

import pandas as pd
from pathlib import Path

from src.experiments.assignment import assign_groups
from src.experiments.build_user_day_retention import build_user_day_retention
from src.experiments.simulate_lift import apply_retention_lift
from src.experiments.analyze import test_proportions
from src.experiments.registry import log_experiment_run

EVENTS_PARQUET = Path("data/processed/events_20201101_20201130.parquet")
USERS_PARQUET = Path("data/processed/users_20201101_20201130.parquet")


def main() -> None:
    # Load minimal columns needed
    events = pd.read_parquet(EVENTS_PARQUET)[["user_id", "event_date"]]
    users = pd.read_parquet(USERS_PARQUET)[["user_id"]]

    # Build user-level D7 retention observation table
    d7 = build_user_day_retention(events, day_n=7)

    # Assign groups (only for users eligible for D7 observation)
    eligible_users = d7[["user_id"]].drop_duplicates()
    groups = assign_groups(eligible_users, seed=42)

    df = d7.merge(groups, on="user_id", how="inner")

    # Simulate a +0.5 percentage point absolute lift in treatment
    lift_abs = 0.005
    seed_sim = 123
    df_sim = apply_retention_lift(df, lift_abs=lift_abs, seed=seed_sim)

    results = test_proportions(df_sim)

    print("Experiment results (D7 retention):")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Log experiment to registry
    meta = {
        "experiment_id": "exp_001",
        "experiment_name": "onboarding_d7_retention_lift",
        "metric": "d7_retention",
        "lift_abs_simulated": lift_abs,
        "seed_assignment": 42,
        "seed_simulation": seed_sim,
        "data_window": "20201101-20201130",
    }
    log_experiment_run(results, meta)
    print("Logged run to logs/experiment_registry.csv")


if __name__ == "__main__":
    main()