import pandas as pd
from pathlib import Path

EVENTS_PARQUET = Path("data/processed/events_20201101_20201130.parquet")
OUT_USERS = Path("data/processed/users_20201101_20201130.parquet")

def mode(series: pd.Series):
    """Return most common value (or NA if empty)."""
    series = series.dropna()
    if series.empty:
        return pd.NA
    return series.value_counts().idxmax()

def main() -> None:
    events = pd.read_parquet(EVENTS_PARQUET)

    # Ensure event_date is a string like "20201101"
    events["event_date"] = events["event_date"].astype(str)

    users = (
        events.groupby("user_id", as_index=False)
        .agg(
            first_seen_date=("event_date", "min"),
            country=("country", mode),
            device_category=("device_category", mode),
            platform=("platform", mode),
        )
    )

    OUT_USERS.parent.mkdir(parents=True, exist_ok=True)
    users.to_parquet(OUT_USERS, index=False)

    print(f"Saved {len(users):,} users to {OUT_USERS}")

if __name__ == "__main__":
    main()