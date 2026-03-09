import pandas as pd
from src.metrics.retention import add_days_yyyymmdd

def build_user_day_retention(events: pd.DataFrame, day_n: int) -> pd.DataFrame:
    """
    Returns a user-level table:
    user_id, first_seen_date, target_date, retained (0/1)
    Only includes users where target_date is within dataset range.
    """
    active = events[["user_id", "event_date"]].drop_duplicates()
    active["event_date"] = active["event_date"].astype(str)

    first_seen = (
        active.groupby("user_id", as_index=False)
        .agg(first_seen_date=("event_date", "min"))
    )
    first_seen["target_date"] = first_seen["first_seen_date"].apply(
        lambda d: add_days_yyyymmdd(d, day_n)
    )

    min_date = active["event_date"].min()
    max_date = active["event_date"].max()
    eligible = first_seen[first_seen["target_date"].between(min_date, max_date)]

    merged = eligible.merge(
        active,
        left_on=["user_id", "target_date"],
        right_on=["user_id", "event_date"],
        how="left",
        indicator=True
    )
    merged["retained"] = (merged["_merge"] == "both").astype(int)

    return merged[["user_id", "first_seen_date", "target_date", "retained"]]