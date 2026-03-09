import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("group", as_index=False)
        .agg(n=("retained", "size"), retained=("retained", "sum"))
    )
    summary["rate"] = summary["retained"] / summary["n"]
    return summary

def test_proportions(df: pd.DataFrame) -> dict:
    summary = summarize(df)
    control = summary[summary["group"] == "control"].iloc[0]
    treat = summary[summary["group"] == "treatment"].iloc[0]

    counts = [treat["retained"], control["retained"]]
    nobs = [treat["n"], control["n"]]

    stat, pval = proportions_ztest(counts, nobs)
    lift_abs = treat["rate"] - control["rate"]
    lift_rel = lift_abs / control["rate"] if control["rate"] > 0 else float("nan")

    return {
        "control_rate": float(control["rate"]),
        "treatment_rate": float(treat["rate"]),
        "lift_abs": float(lift_abs),
        "lift_rel": float(lift_rel),
        "p_value": float(pval),
        "n_control": int(control["n"]),
        "n_treatment": int(treat["n"]),
    }
    