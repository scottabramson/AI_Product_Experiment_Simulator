from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Experimentation Lab", layout="wide")

# -----------------------
# Path resolution
# Prefer local/full artifacts when available, otherwise fall back to committed samples
# -----------------------
DAU_FULL = Path("data/processed/dau_20201101_20201130.parquet")
RET_FULL = Path("data/processed/retention_d1_d7_20201101_20201130.parquet")
REG_FULL = Path("logs/experiment_registry.csv")

DAU_SAMPLE = Path("data/sample/dau_sample.parquet")
RET_SAMPLE = Path("data/sample/retention_sample.parquet")
REG_SAMPLE = Path("data/sample/experiment_registry_sample.csv")

DAU_PATH = DAU_FULL if DAU_FULL.exists() else DAU_SAMPLE
RET_PATH = RET_FULL if RET_FULL.exists() else RET_SAMPLE
REG_PATH = REG_FULL if REG_FULL.exists() else REG_SAMPLE

st.title("Product Experimentation Lab")

# -----------------------
# Verify required files exist (resolved paths)
# -----------------------
missing = [str(p) for p in [DAU_PATH, RET_PATH, REG_PATH] if not p.exists()]
if missing:
    st.error("Missing required file(s):")
    for m in missing:
        st.write(f"- {m}")
    st.stop()

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_dau(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Fix Arrow LargeUtf8 issues on Streamlit Cloud (Streamlit 1.19)
    if "event_date" in df.columns:
        df["event_date"] = df["event_date"].astype(str)

    df["date"] = pd.to_datetime(df["event_date"], format="%Y%m%d", errors="coerce")
    return df.sort_values("date")


@st.cache_data
def load_retention(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "first_seen_date" in df.columns:
        df["first_seen_date"] = df["first_seen_date"].astype(str)

    df["cohort_date"] = pd.to_datetime(df["first_seen_date"], format="%Y%m%d", errors="coerce")
    return df.sort_values(["day_n", "cohort_date"])


@st.cache_data
def load_registry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_utc"] = pd.to_datetime(df["run_utc"], errors="coerce")
    return df.sort_values("run_utc", ascending=False)


dau = load_dau(DAU_PATH)
ret = load_retention(RET_PATH)
reg = load_registry(REG_PATH)

# -----------------------
# Top KPIs
# -----------------------
col1, col2, col3 = st.columns(3)
col1.metric("Days in dataset", len(dau))
col2.metric("Total events (window)", int(dau["events"].sum()))
col3.metric("Avg DAU", int(dau["dau"].mean()))

st.caption(f"Data source: {'full (local)' if DAU_PATH == DAU_FULL else 'sample (repo)'}")

st.markdown("---")
# -----------------------
# DAU chart
# -----------------------
st.subheader("Daily Active Users (DAU)")
st.line_chart(dau.set_index("date")[["dau"]])

# -----------------------
# Retention chart
# -----------------------
st.subheader("Cohort Retention (D1 vs D7)")
d1 = ret[ret["day_n"] == 1][["cohort_date", "retention_rate"]].rename(columns={"retention_rate": "D1"})
d7 = ret[ret["day_n"] == 7][["cohort_date", "retention_rate"]].rename(columns={"retention_rate": "D7"})
ret_plot = pd.merge(d1, d7, on="cohort_date", how="outer").sort_values("cohort_date")

st.line_chart(ret_plot.set_index("cohort_date")[["D1", "D7"]])

st.markdown("---")
# -----------------------
# Experiment registry
# -----------------------
st.subheader("Experiment Registry")

if reg.empty:
    st.info("No experiment runs logged yet. Run: python -m src.prepare.run_experiment_d7")
else:
    latest = reg.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest experiment", str(latest.get("experiment_id", "")))
    c2.metric("Control rate", f"{float(latest.get('control_rate', 0.0)):.4f}")
    c3.metric("Treatment rate", f"{float(latest.get('treatment_rate', 0.0)):.4f}")
    c4.metric("Lift (abs)", f"{float(latest.get('lift_abs', 0.0)):.4f}")

st.dataframe(reg.astype(str), use_container_width=True)