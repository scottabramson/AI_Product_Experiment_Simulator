import pandas as pd
from pathlib import Path

RAW_CSV = Path("data/raw/events_20201101_20201130.csv")
OUT_PARQUET = Path("data/processed/events_20201101_20201130.parquet")

def main() -> None:
    # 1) Load CSV
    df = pd.read_csv(RAW_CSV)

    # 2) Basic cleanup / types
    # event_date is usually like 20201101 (string or int). Keep as string for now.
    df["event_date"] = df["event_date"].astype(str)

    # event_time should be parseable datetime (we exported as timestamp)
    # If your column is named event_time, this will work:
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

    # 3) Save as parquet (fast)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"Saved {len(df):,} rows to {OUT_PARQUET}")

if __name__ == "__main__":
    main()