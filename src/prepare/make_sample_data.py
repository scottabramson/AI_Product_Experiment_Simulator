from pathlib import Path
import pandas as pd

# Full (local) artifacts
DAU_FULL = Path("data/processed/dau_20201101_20201130.parquet")
RET_FULL = Path("data/processed/retention_d1_d7_20201101_20201130.parquet")
REG_FULL = Path("logs/experiment_registry.csv")

# Sample artifacts (commit these)
SAMPLE_DIR = Path("data/sample")
DAU_SAMPLE = SAMPLE_DIR / "dau_sample.parquet"
RET_SAMPLE = SAMPLE_DIR / "retention_sample.parquet"
REG_SAMPLE = SAMPLE_DIR / "experiment_registry_sample.csv"

# Pick a small window that still supports D7 retention cohorts
# If your data includes 20201101-20201130, this is safe.
SAMPLE_START = "20201101"
SAMPLE_END = "20201114"  # 14 days gives enough runway to illustrate metrics


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # --- DAU sample (filter by date window) ---
    dau = pd.read_parquet(DAU_FULL)
    dau["event_date"] = dau["event_date"].astype(str)
    dau_s = dau[dau["event_date"].between(SAMPLE_START, SAMPLE_END)].copy()
    dau_s.to_parquet(DAU_SAMPLE, index=False)
    print(f"Saved DAU sample -> {DAU_SAMPLE} ({len(dau_s)} rows)")

    # --- Retention sample (filter cohorts by first_seen_date window) ---
    ret = pd.read_parquet(RET_FULL)
    ret["first_seen_date"] = ret["first_seen_date"].astype(str)
    ret_s = ret[ret["first_seen_date"].between(SAMPLE_START, SAMPLE_END)].copy()
    ret_s.to_parquet(RET_SAMPLE, index=False)
    print(f"Saved retention sample -> {RET_SAMPLE} ({len(ret_s)} rows)")

    # --- Experiment registry sample (keep last ~10 rows) ---
    if REG_FULL.exists():
        reg = pd.read_csv(REG_FULL)
        reg_s = reg.tail(10).copy()
        reg_s.to_csv(REG_SAMPLE, index=False)
        print(f"Saved registry sample -> {REG_SAMPLE} ({len(reg_s)} rows)")
    else:
        # Create an empty but valid CSV so the app can still run
        empty_cols = [
            "experiment_id", "experiment_name", "metric", "lift_abs_simulated",
            "seed_assignment", "seed_simulation", "data_window",
            "control_rate", "treatment_rate", "lift_abs", "lift_rel",
            "p_value", "n_control", "n_treatment", "run_utc"
        ]
        pd.DataFrame(columns=empty_cols).to_csv(REG_SAMPLE, index=False)
        print(f"Saved empty registry sample -> {REG_SAMPLE}")

    print("\nDone. Commit data/sample/* to GitHub so Streamlit Cloud can run.")


if __name__ == "__main__":
    main()