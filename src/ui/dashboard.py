import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Allow imports when run as: streamlit run src/ui/dashboard.py
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.repo import SqliteRepo
from ui.caster_monitor import (
    CasterContext,
    aggregate_metrics,
    fetch_recent_pipes,
    get_all_casters,
    health_rows,
    read_log_tail,
)
from ui.formatting import fmt_ts
from utils.config import resolve_caster_id


ALL_CASTERS = "All Casters"


st.set_page_config(layout="wide", page_title="Caster Monitoring Dashboard")
st.title("Caster Monitoring Dashboard")


@st.cache_data(ttl=10)
def _load_casters() -> list[CasterContext]:
    return get_all_casters()


def _initial_selection(options: list[str]) -> str:
    env_caster = os.environ.get("PIPE_DASHBOARD_CASTER")
    if not env_caster:
        return ALL_CASTERS
    try:
        key = f"caster_{resolve_caster_id(env_caster)}"
    except ValueError:
        return ALL_CASTERS
    return key if key in options else ALL_CASTERS


def _metric_row(metrics) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last hour", metrics.last_hour)
    c2.metric("Last 8h", metrics.last_8h)
    c3.metric("Last 24h", metrics.last_24h)
    c4.metric("Avg conf 8h", f"{metrics.avg_conf_8h:.2f}")


def _camera_grid(contexts: list[CasterContext], columns: int = 2) -> None:
    st.subheader("Connected Cameras")
    if not contexts:
        st.info("No configured casters found.")
        return

    for start in range(0, len(contexts), columns):
        cols = st.columns(columns)
        for col, ctx in zip(cols, contexts[start : start + columns]):
            with col:
                status = health_rows([ctx])[0]
                camera_type = ctx.cfg.camera_cfg.type if ctx.cfg.camera_cfg else "opencv"
                fps = ctx.cfg.camera_cfg.fps if ctx.cfg.camera_cfg else "-"
                frame_path = ctx.latest_frame_path

                st.markdown(f"**{ctx.caster_key}**")
                st.caption(f"{camera_type} | {status['Status']} | fps={fps} | {status['Frame Age']}")
                if frame_path.exists():
                    st.image(str(frame_path), use_column_width=True)
                else:
                    st.info("Waiting for latest frame")


def _health_table(contexts: list[CasterContext]) -> None:
    st.subheader("Caster Health")
    rows = health_rows(contexts)
    if not rows:
        st.info("No configured casters found.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _pipes_dataframe(ctx: CasterContext) -> pd.DataFrame:
    rows = fetch_recent_pipes(ctx, limit=250)
    df = pd.DataFrame(
        rows,
        columns=[
            "pipe_uid",
            "origin",
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
    for col in ("t_origin", "t_loadcell_enter", "t_loadcell_exit", "last_seen_ts"):
        if col in df.columns:
            df[col] = df[col].apply(fmt_ts)
    for col in ("pipe_uid", "origin", "state", "t_origin", "t_loadcell_enter", "t_loadcell_exit", "last_seen_ts"):
        if col in df.columns:
            df[col] = df[col].map(lambda value: "" if value is None else str(value)).astype(object)
    return df


def _single_caster_view(ctx: CasterContext) -> None:
    metrics = aggregate_metrics([ctx])
    _metric_row(metrics)

    tab_live, tab_pipes, tab_config, tab_logs = st.tabs(["Live", "Recent Pipes", "Configuration", "Logs"])

    with tab_live:
        _health_table([ctx])
        _camera_grid([ctx], columns=1)

    with tab_pipes:
        st.subheader(f"Recent Pipes - {ctx.caster_key}")
        st.dataframe(_pipes_dataframe(ctx), use_container_width=True, hide_index=True)

    with tab_config:
        st.subheader("Runtime Context")
        camera_cfg = ctx.cfg.camera_cfg
        st.json(
            {
                "caster": ctx.caster_key,
                "camera_type": camera_cfg.type if camera_cfg else "opencv",
                "video_source": ctx.cfg.runtime.video_source,
                "db_path": str(ctx.db_path),
                "latest_frame": str(ctx.latest_frame_path),
                "log_path": str(ctx.log_path) if ctx.log_path else None,
                "plc_mode": ctx.cfg.plc.mode,
                "model_path": ctx.cfg.runtime.model_path,
            }
        )

    with tab_logs:
        st.subheader(f"Logs - {ctx.caster_key}")
        tail = read_log_tail(ctx)
        st.code(tail or "No log file found.", language="text")


def _all_casters_view(contexts: list[CasterContext]) -> None:
    metrics = aggregate_metrics(contexts)
    _metric_row(metrics)
    _health_table(contexts)
    _camera_grid(contexts, columns=2)


casters = _load_casters()
caster_by_key = {ctx.caster_key: ctx for ctx in casters}
options = [ALL_CASTERS, *caster_by_key.keys()]

with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Caster", options, index=options.index(_initial_selection(options)))
    refresh_ms = st.slider("Auto-refresh interval (ms)", 1000, 10000, 5000, 1000)

    selected_ctx = caster_by_key.get(selected)
    gate_source = st.selectbox("Gate source", ["geometry", "plc", "vision"], disabled=selected_ctx is None)
    if st.button("Apply gate source", disabled=selected_ctx is None):
        SqliteRepo(str(selected_ctx.db_path)).set_setting("gate_source", gate_source)
        st.success(f"{selected_ctx.caster_key} gate source set to {gate_source}")

st_autorefresh(interval=refresh_ms, key="dashboard_refresh")

if selected == ALL_CASTERS:
    _all_casters_view(casters)
else:
    _single_caster_view(caster_by_key[selected])
