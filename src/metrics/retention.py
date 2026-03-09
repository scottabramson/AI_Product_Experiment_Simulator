import pandas as pd

def add_days_yyyymmdd(date_str: str, days: int) -> str:
    dt = pd.to_datetime(date_str, format="%Y%m%d")
    return (dt + pd.Timedelta(days=days)).strftime("%Y%m%d")

def compute_retention(events: pd.DataFrame, day_n: int) -> pd.DataFrame:
    # active days per user (dedupe so 100 events in a day still counts as "active once")
    active = events[["user_id", "event_date"]].drop_duplicates()
    active["event_date"] = active["event_date"].astype(str)

    # first day each user was seen
    first_seen = (
        active.groupby("user_id", as_index=False)
        .agg(first_seen_date=("event_date", "min"))
    )

    # target date = first_seen + N days
    first_seen["target_date"] = first_seen["first_seen_date"].apply(
        lambda d: add_days_yyyymmdd(d, day_n)
    )

    # only include users where we can actually observe their day N in this dataset
    min_date = active["event_date"].min()
    max_date = active["event_date"].max()
    eligible = first_seen[first_seen["target_date"].between(min_date, max_date)]

    # check if the user is active on target_date
    merged = eligible.merge(
        active,
        left_on=["user_id", "target_date"],
        right_on=["user_id", "event_date"],
        how="left",
        indicator=True
    )

    merged["retained"] = (merged["_merge"] == "both").astype(int)

    # retention by cohort (first_seen_date)
    out = (
        merged.groupby("first_seen_date", as_index=False)
        .agg(
            cohort_size=("user_id", "nunique"),
            retained=("retained", "sum")
        )
    )
    out["retention_rate"] = out["retained"] / out["cohort_size"]
    out["day_n"] = day_n
    return out