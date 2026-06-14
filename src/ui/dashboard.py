import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------
# Allow imports when run as: streamlit run src/ui/dashboard.py
# -------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.repo import SqliteRepo
from ui.formatting import fmt_ts

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DB_PATH = "var/pipes.db"
FRAMEPATH = "var/latest.jpg"

st.set_page_config(
    layout="wide",
    page_title="Pipe Tracking Dashboard",
)

st.title("Pipe Tracking Dashboard (Local)")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Controls")

    gate_source = st.selectbox(
        "Gate source",
        ["geometry", "plc (not implemented)", "vision (not implemented)"],
    )

    if st.button("Apply gate source"):
        SqliteRepo(DB_PATH).set_setting("gate_source", gate_source)
        st.success(f"Gate source set to {gate_source}")

    refresh_ms = st.slider(
        "Auto-refresh interval (ms)",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=1000,
    )

# -------------------------------------------------
# AUTO REFRESH (Streamlit-native)
# -------------------------------------------------
st_autorefresh(interval=refresh_ms, key="dashboard_refresh")

# -------------------------------------------------
# DB (single instance per rerun)
# -------------------------------------------------
repo = SqliteRepo(DB_PATH)

# -------------------------------------------------
# METRICS (TOP)
# -------------------------------------------------
st.header("Pipes Casted")
m1, m2, m3 = st.columns(3)

m1.metric("Last hour", repo.metric_counts(3600))
m2.metric("Last 8h", repo.metric_counts(8 * 3600))
m3.metric("Last 24h", repo.metric_counts(24 * 3600))

st.markdown("---")  # horizontal divider

# -------------------------------------------------
# IMAGE (MIDDLE)
# -------------------------------------------------
st.subheader("Latest Annotated Frame")

try:
    st.image(
        FRAMEPATH,
        use_column_width=True,
        caption="Latest annotated frame",
    )
except Exception as e:
    st.info(f"Waiting for image… ({e})")

st.markdown("---")  # horizontal divider

# -------------------------------------------------
# TABLE (BOTTOM)
# -------------------------------------------------
st.subheader("Recent Pipes")

rows = repo.fetch_pipes(limit=250)

df = pd.DataFrame(
    rows,
    columns=[
        "pipe_uid",
        "origin",
        "pipe_checkpoint",
        "t_origin",
        "t_loadcell_enter",
        "t_loadcell_exit",
        "weight",
        "weight_quality",
        "weight_samples",
        "avg_conf_full",
        "avg_conf_till_gate",
        "frames_missing",
        "state",
        "last_seen_ts",
    ],
)

# Format timestamps
for c in (
    "t_origin",
    "t_loadcell_enter",
    "t_loadcell_exit",
    "last_seen_ts",
):
    if c in df.columns:
        df[c] = df[c].apply(fmt_ts)

# Ensure Arrow-safe text columns
for c in (
    "pipe_uid",
    "origin",
    "state",
    "t_origin",
    "t_loadcell_enter",
    "t_loadcell_exit",
    "last_seen_ts",
):
    if c in df.columns:
        df[c] = df[c].map(lambda v: "" if v is None else str(v)).astype(object)

st.dataframe(df)
