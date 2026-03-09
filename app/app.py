from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Experimentation Lab", layout="wide")


def make_streamlit_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Arrow-backed / extension dtypes into plain Python-friendly types
    so Streamlit can render them safely.
    """
    df = df.copy()

    # Make sure column names are plain strings
    df.columns = [str(c) for c in df.columns]

    for col in df.columns:
        dtype_str = str(df[col].dtype)

        # Convert problematic string-like dtypes to plain object strings
        if (
            dtype_str in ("string", "string[pyarrow]", "large_string[pyarrow]")
            or "string" in dtype_str.lower()
            or "utf8" in dtype_str.lower()
        ):
            df[col] = df[col].astype(str).astype(object)

    return df


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

    # Normalize all string-like columns early
    df = make_streamlit_safe(df)

    if "event_date" in df.columns:
        df["event_date"] = df["event_date"].astype(str).astype(object)

    df["date"] = pd.to_datetime(df["event_date"], format="%Y%m%d", errors="coerce")
    return df.sort_values("date")


@st.cache_data
def load_retention(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Normalize all string-like columns early
    df = make_streamlit_safe(df)

    if "first_seen_date" in df.columns:
        df["first_seen_date"] = df["first_seen_date"].astype(str).astype(object)

    df["cohort_date"] = pd.to_datetime(df["first_seen_date"], format="%Y%m%d", errors="coerce")
    return df.sort_values(["day_n", "cohort_date"])


@st.cache_data
def load_registry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize all string-like columns early
    df = make_streamlit_safe(df)

    if "run_utc" in df.columns:
        df["run_utc"] = pd.to_datetime(df["run_utc"], errors="coerce")

    return df.sort_values("run_utc", ascending=False)


dau = load_dau(DAU_PATH)
ret = load_retention(RET_PATH)
reg = load_registry(REG_PATH)

# Make extra sure displayed frames are safe
dau = make_streamlit_safe(dau)
ret = make_streamlit_safe(ret)
reg = make_streamlit_safe(reg)

# -----------------------
# Top KPIs
# -----------------------
col1, col2, col3 = st.columns(3)
col1.metric("Days in dataset", len(dau))
col2.metric("Total events (window)", int(pd.to_numeric(dau["events"], errors="coerce").fillna(0).sum()))
col3.metric("Avg DAU", int(pd.to_numeric(dau["dau"], errors="coerce").fillna(0).mean()))

st.caption(f"Data source: {'full (local)' if DAU_PATH == DAU_FULL else 'sample (repo)'}")

st.markdown("---")

# -----------------------
# DAU chart
# -----------------------
st.subheader("Daily Active Users (DAU)")
dau_chart = dau[["date", "dau"]].copy()
dau_chart["dau"] = pd.to_numeric(dau_chart["dau"], errors="coerce")
dau_chart = make_streamlit_safe(dau_chart)
st.line_chart(dau_chart.set_index("date")[["dau"]])

# -----------------------
# Retention chart
# -----------------------
st.subheader("Cohort Retention (D1 vs D7)")
d1 = ret[ret["day_n"] == 1][["cohort_date", "retention_rate"]].rename(columns={"retention_rate": "D1"})
d7 = ret[ret["day_n"] == 7][["cohort_date", "retention_rate"]].rename(columns={"retention_rate": "D7"})
ret_plot = pd.merge(d1, d7, on="cohort_date", how="outer").sort_values("cohort_date")

ret_plot["D1"] = pd.to_numeric(ret_plot["D1"], errors="coerce")
ret_plot["D7"] = pd.to_numeric(ret_plot["D7"], errors="coerce")
ret_plot = make_streamlit_safe(ret_plot)

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
    c2.metric("Control rate", f"{float(pd.to_numeric(latest.get('control_rate', 0.0), errors='coerce')):.4f}")
    c3.metric("Treatment rate", f"{float(pd.to_numeric(latest.get('treatment_rate', 0.0), errors='coerce')):.4f}")
    c4.metric("Lift (abs)", f"{float(pd.to_numeric(latest.get('lift_abs', 0.0), errors='coerce')):.4f}")

reg_display = make_streamlit_safe(reg.astype(str))
st.dataframe(reg_display, use_container_width=True)