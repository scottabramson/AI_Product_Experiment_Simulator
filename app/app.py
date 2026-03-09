from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Experimentation Lab", layout="wide")


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string-like extension dtypes to plain Python strings."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    for col in df.columns:
        dtype_str = str(df[col].dtype).lower()
        if "string" in dtype_str or "utf8" in dtype_str:
            df[col] = df[col].astype(str)

    return df


# -----------------------
# Path resolution
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
# Verify required files exist
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
    df = clean_strings(df)

    if "event_date" in df.columns:
        df["event_date"] = df["event_date"].astype(str)

    df["date"] = pd.to_datetime(df["event_date"], format="%Y%m%d", errors="coerce")
    df["dau"] = pd.to_numeric(df["dau"], errors="coerce")
    df["events"] = pd.to_numeric(df["events"], errors="coerce")
    return df.sort_values("date")


@st.cache_data
def load_retention(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = clean_strings(df)

    if "first_seen_date" in df.columns:
        df["first_seen_date"] = df["first_seen_date"].astype(str)

    df["cohort_date"] = pd.to_datetime(df["first_seen_date"], format="%Y%m%d", errors="coerce")
    df["day_n"] = pd.to_numeric(df["day_n"], errors="coerce")
    df["retention_rate"] = pd.to_numeric(df["retention_rate"], errors="coerce")
    return df.sort_values(["day_n", "cohort_date"])


@st.cache_data
def load_registry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_strings(df)

    if "run_utc" in df.columns:
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
col2.metric("Total events (window)", int(dau["events"].fillna(0).sum()))
col3.metric("Avg DAU", int(dau["dau"].fillna(0).mean()))

st.caption(f"Data source: {'full (local)' if DAU_PATH == DAU_FULL else 'sample (repo)'}")

st.markdown("---")

# -----------------------
# DAU chart
# -----------------------
st.subheader("Daily Active Users (DAU)")
dau_chart = dau[["date", "dau"]].dropna().sort_values("date")

fig1, ax1 = plt.subplots()
ax1.plot(dau_chart["date"], dau_chart["dau"])
ax1.set_xlabel("Date")
ax1.set_ylabel("DAU")
ax1.set_title("Daily Active Users")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# -----------------------
# Retention chart
# -----------------------
st.subheader("Cohort Retention (D1 vs D7)")
d1 = ret[ret["day_n"] == 1][["cohort_date", "retention_rate"]].rename(columns={"retention_rate": "D1"})
d7 = ret[ret["day_n"] == 7][["cohort_date", "retention_rate"]].rename(columns={"retention_rate": "D7"})
ret_plot = pd.merge(d1, d7, on="cohort_date", how="outer").sort_values("cohort_date")

fig2, ax2 = plt.subplots()
ax2.plot(ret_plot["cohort_date"], ret_plot["D1"], label="D1")
ax2.plot(ret_plot["cohort_date"], ret_plot["D7"], label="D7")
ax2.set_xlabel("Cohort Date")
ax2.set_ylabel("Retention Rate")
ax2.set_title("Cohort Retention")
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

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
    c2.metric("Control rate", f"{float(pd.to_numeric(latest.get('control_rate', 0.0), errors='coerce')):.4f}")
    c3.metric("Treatment rate", f"{float(pd.to_numeric(latest.get('treatment_rate', 0.0), errors='coerce')):.4f}")
    c4.metric("Lift (abs)", f"{float(pd.to_numeric(latest.get('lift_abs', 0.0), errors='coerce')):.4f}")

    reg_display = reg.copy().head(10)
    reg_display = reg_display.astype(str)

    st.markdown(
        reg_display.to_html(index=False, escape=False),
        unsafe_allow_html=True
    )