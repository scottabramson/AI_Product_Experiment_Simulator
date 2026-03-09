from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

REGISTRY_PATH = Path("logs/experiment_registry.csv")

def log_experiment_run(result: dict, meta: dict) -> None:
    row = {**meta, **result}
    row["run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    df_row = pd.DataFrame([row])
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if REGISTRY_PATH.exists():
        df_existing = pd.read_csv(REGISTRY_PATH)
        df_out = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_out = df_row

    df_out.to_csv(REGISTRY_PATH, index=False)
    