import numpy as np
import pandas as pd

def apply_retention_lift(df: pd.DataFrame, lift_abs: float, seed: int = 123) -> pd.DataFrame:
    """
    Apply an absolute lift to treatment users by flipping some 0s to 1s.
    lift_abs = 0.005 means +0.5 percentage points absolute retention.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    # Only treatment users who are NOT retained can be flipped to retained
    mask = (out["group"] == "treatment") & (out["retained"] == 0)
    candidates = out[mask].index

    # How many flips needed? Based on treatment size.
    treat_n = (out["group"] == "treatment").sum()
    flips = int(round(lift_abs * treat_n))

    if flips > len(candidates):
        flips = len(candidates)

    if flips > 0:
        chosen = rng.choice(candidates, size=flips, replace=False)
        out.loc[chosen, "retained"] = 1

    return out