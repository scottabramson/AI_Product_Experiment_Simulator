import pandas as pd
from pathlib import Path

EVENTS_PARQUET = Path("data/processed/events_20201101_20201130.parquet")
OUT_DAU = Path("data/processed/dau_20201101_20201130.parquet")

def main() -> None:
    events = pd.read_parquet(EVENTS_PARQUET)

    # Only keep what we need (faster)
    events = events[["user_id", "event_date", "event_name"]]
    events["event_date"] = events["event_date"].astype(str)

    dau = (
        events.groupby("event_date", as_index=False)
        .agg(
            dau=("user_id", "nunique"),
            events=("event_name", "size"),
        )
        .sort_values("event_date")
    )

    OUT_DAU.parent.mkdir(parents=True, exist_ok=True)
    dau.to_parquet(OUT_DAU, index=False)
    print(f"Saved {len(dau):,} days -> {OUT_DAU}")
    print(dau.head())

if __name__ == "__main__":
    main()