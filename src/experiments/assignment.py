import numpy as np
import pandas as pd

def assign_groups(users: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign each user to control or treatment with a 50/50 split.
    Seeded so results are reproducible.
    """
    rng = np.random.default_rng(seed)
    out = users[["user_id"]].copy()
    out["group"] = rng.choice(["control", "treatment"], size=len(out))
    return out
