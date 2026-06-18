import sys
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.repo import SqliteRepo
from ui.formatting import fmt_ts

DB_PATH = "var/pipes.db"
FRAMEPATH = "var/latest.jpg"
REFRESH_SECONDS = 5
DEVELOPER_TABLES = (
    "events",
    "settings",
    "gate_open_events",
    "loadcell_events",
)
DEVELOPER_TABLE_ORDER = {
    "events": "COALESCE(ts, id)",
    "settings": "updated_at",
    "gate_open_events": "COALESCE(t_open, id)",
    "loadcell_events": "COALESCE(created_at, t_enter, t_exit, id)",
}
DEVELOPER_TIME_COLUMNS = (
    "ts",
    "updated_at",
    "t_open",
    "t_enter",
    "t_exit",
    "created_at",
)

st.set_page_config(layout="wide", page_title="Pipe Tracking Dashboard")

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 20px;
}
.block-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Pipe Tracking Dashboard</div>', unsafe_allow_html=True)


def repo():
    return SqliteRepo(DB_PATH)


def set_gate_source(value: str):
    r = repo()
    if hasattr(r, "set_setting"):
        r.set_setting("gate_source", value)
    else:
        r.setting("gate_source", value)


def build_recent_df(limit=250):
    rows = repo().fetch_pipes(limit=limit)

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

    for c in ("t_origin", "t_loadcell_enter", "t_loadcell_exit", "last_seen_ts"):
        if c in df.columns:
            df[c] = df[c].apply(fmt_ts)

    text_cols = (
        "pipe_uid",
        "origin",
        "pipe_checkpoint",
        "weight_quality",
        "state",
        "t_origin",
        "t_loadcell_enter",
        "t_loadcell_exit",
        "last_seen_ts",
    )

    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: "" if v is None else str(v)).astype(object)

    return df


def read_table(table_name, limit=500):
    if table_name not in DEVELOPER_TABLES:
        raise ValueError(f"Unsupported developer table: {table_name}")

    order_expr = DEVELOPER_TABLE_ORDER[table_name]
    query = f'SELECT * FROM "{table_name}" ORDER BY {order_expr} DESC LIMIT ?'

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=(limit,))


def format_developer_df(df):
    df = df.copy()

    for c in DEVELOPER_TIME_COLUMNS:
        if c in df.columns:
            df[c] = df[c].apply(fmt_ts)
            df[c] = df[c].map(lambda v: "" if v is None else str(v)).astype(object)

    return df


with st.sidebar:
    st.header("Controls")

    gate_source = st.selectbox(
        "Gate source",
        ["geometry", "plc (not implemented)", "vision (not implemented)"],
    )

    if st.button("Apply gate source", width="stretch"):
        set_gate_source(gate_source)
        st.success(f"Gate source set to {gate_source}")

    REFRESH_SECONDS = st.slider(
        "Refresh interval",
        min_value=1,
        max_value=10,
        value=REFRESH_SECONDS,
        step=1,
        format="%d sec",
    )

    developer_mode = st.toggle("Developer Mode", value=False)


@st.fragment(run_every=REFRESH_SECONDS)
def live_frame():
    with st.container(border=True):
        st.markdown('<div class="block-title">Latest Annotated Frame</div>', unsafe_allow_html=True)

        frame = Path(FRAMEPATH)
        if frame.exists():
            st.image(str(frame), width="stretch", caption="Latest annotated frame")
        else:
            st.info("Waiting for latest annotated frame...")


@st.fragment(run_every=REFRESH_SECONDS)
def live_metrics():
    with st.container(border=True):
        st.markdown('<div class="block-title">Pipes Casted Data</div>', unsafe_allow_html=True)

        r = repo()

        c1, c2, c3 = st.columns(3)
        c1.metric("Last hour", r.metric_counts(3600))
        c2.metric("Last 8h", r.metric_counts(8 * 3600))
        c3.metric("Last 24h", r.metric_counts(24 * 3600))


@st.fragment(run_every=REFRESH_SECONDS)
def recent_pipes_table():
    st.markdown("### Recent Pipes")

    df = build_recent_df(limit=250)

    if df.empty:
        st.info("No recent pipes found.")
        return

    st.dataframe(
        df,
        width="stretch",
        height=420,
        hide_index=True,
    )


@st.fragment(run_every=REFRESH_SECONDS)
def developer_tables():
    st.markdown("### Developer Tables")

    for table_name in DEVELOPER_TABLES:
        with st.container(border=True):
            st.markdown(f'<div class="block-title">{table_name}</div>', unsafe_allow_html=True)

            try:
                df = read_table(table_name, limit=500)
            except Exception as exc:
                st.error(f"Could not read {table_name}: {exc}")
                continue

            df = format_developer_df(df)

            st.metric("Rows shown", len(df))
            st.dataframe(
                df,
                width="stretch",
                height=300,
                hide_index=True,
            )


left, right = st.columns([1.35, 1], gap="large")

with left:
    live_frame()

with right:
    live_metrics()

st.divider()
recent_pipes_table()

if developer_mode:
    developer_tables()
