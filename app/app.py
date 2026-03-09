import streamlit as st
st.title("DASHBOARD LOADED")
import traceback
from pathlib import Path

import pandas as pd

DAU_PATH = Path("data/processed/dau_20201101_20201130.parquet")
RET_PATH = Path("data/processed/retention_d1_d7_20201101_20201130.parquet")
REG_PATH = Path("logs/experiment_registry.csv")


def main() -> None:
    st.set_page_config(page_title="Experimentation Lab", layout="wide")
    st.title("Product Experimentation Lab")

    # Verify required files exist
    missing = [str(p) for p in [DAU_PATH, RET_PATH, REG_PATH] if not p.exists()]
    if missing:
        st.error("Missing required file(s):")
        for m in missing:
            st.write(f"- {m}")
        st.stop()

    @st.cache_data
    def load_dau():
        df = pd.read_parquet(DAU_PATH)
        df["date"] = pd.to_datetime(df["event_date"], format="%Y%m%d")
        return df.sort_values("date")

    @st.cache_data
    def load_retention():
        df = pd.read_parquet(RET_PATH)
        df["cohort_date"] = pd.to_datetime(df["first_seen_date"], format="%Y%m%d")
        return df.sort_values(["day_n", "cohort_date"])

    @st.cache_data
    def load_registry():
        df = pd.read_csv(REG_PATH)
        df["run_utc"] = pd.to_datetime(df["run_utc"], errors="coerce")
        return df.sort_values("run_utc", ascending=False)

    dau = load_dau()
    ret = load_retention()
    reg = load_registry()

    # Top KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Days in dataset", len(dau))
    col2.metric("Total events (month)", int(dau["events"].sum()))
    col3.metric("Avg DAU", int(dau["dau"].mean()))

    st.divider()

    # DAU chart
    st.subheader("Daily Active Users (DAU)")
    st.line_chart(dau.set_index("date")[["dau"]])

    # Retention chart
    st.subheader("Cohort Retention (D1 vs D7)")
    d1 = ret[ret["day_n"] == 1][["cohort_date", "retention_rate"]].rename(columns={"retention_rate": "D1"})
    d7 = ret[ret["day_n"] == 7][["cohort_date", "retention_rate"]].rename(columns={"retention_rate": "D7"})
    ret_plot = pd.merge(d1, d7, on="cohort_date", how="outer").sort_values("cohort_date")

    st.line_chart(ret_plot.set_index("cohort_date")[["D1", "D7"]])

    st.divider()

    # Experiment registry
    st.subheader("Experiment Registry")
    if reg.empty:
        st.info("No experiment runs logged yet. Run: python -m src.prepare.run_experiment_d7")
    else:
        latest = reg.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest experiment", str(latest["experiment_id"]))
        c2.metric("Control rate", f"{latest['control_rate']:.4f}")
        c3.metric("Treatment rate", f"{latest['treatment_rate']:.4f}")
        c4.metric("Lift (abs)", f"{latest['lift_abs']:.4f}")

        st.dataframe(reg, use_container_width=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # This guarantees you’ll see the error in the app instead of a blank screen
        st.error("App crashed. Here is the traceback:")
        st.code(traceback.format_exc())