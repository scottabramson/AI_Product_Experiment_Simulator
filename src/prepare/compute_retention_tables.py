import pandas as pd
from pathlib import Path
from src.metrics.retention import compute_retention

EVENTS_PARQUET = Path("data/processed/events_20201101_20201130.parquet")
OUT_RET = Path("data/processed/retention_d1_d7_20201101_20201130.parquet")

def main() -> None:
    events = pd.read_parquet(EVENTS_PARQUET)

    # Keep only needed columns (faster)
    events = events[["user_id", "event_date"]]
    events["event_date"] = events["event_date"].astype(str)

    d1 = compute_retention(events, 1)
    d7 = compute_retention(events, 7)

    out = pd.concat([d1, d7], ignore_index=True)

    OUT_RET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_RET, index=False)

    print(f"Saved retention table -> {OUT_RET}")
    print(out.head(10))

if __name__ == "__main__":
    main()