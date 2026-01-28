import sys
from pathlib import Path
import time
from datetime import datetime

import pandas as pd
import streamlit as st

# allow imports when run as: streamlit run src/ui/dashboard.py
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from db.repo import SqliteRepo

#TODO: should be in config
DB_PATH = "var/pipes.db"
FRAMEPATH = "var/latest.jpg"


def fmt_ts(x) -> str:
  if x is None:
    return ""
  try: 
    return datetime.fromtimestamp(float(x)).strftime("%Y-%m-%d %H:%M:%S")
  except Exception:
    return ""

st.set_page_config(layout="wide", page_title="Pipe Tracking Dashboard")
st.title("Pipe Tracking Dashboard (Local)")

repo = SqliteRepo(DB_PATH)

with st.sidebar:
  st.header("Controls")
  gate_source = st.selectbox("Gate source", ["geometry", "plc (not implemented)", "vision (not implemented)"])
  if st.button("Apply gate source"):
    repo.setting("gate_source", gate_source)
    st.success(f"Gate source set to {gate_source}")

  refresh_ms = st.slider("Auto-refresh interval (ms)", min_value=1000, max_value=10000, value=5000, step=1000)

# Metrics
c1, c2, c3 = st.columns(3)
st.header("Pipes Casted")
c1.metric("Last hour", repo.metric_counts(3600))
c2.metric("Last 8h", repo.metric_counts(8 * 3600))
c3.metric("Last 24h", repo.metric_counts(24 * 3600))

a1, a2, a3 = st.columns(3)
st.header("Avg Detection Confidence")
a1.metric("Last hour", f"{repo.metric_avg_conf(3600):.3f}")
a2.metric("Last 8h", f"{repo.metric_avg_conf(8 * 3600):.3f}")
a3.metric("Last 24h", f"{repo.metric_avg_conf(24 * 3600):.3f}")

left = st.columns(1) # just a placeholder
right = st.columns(1) # just a placeholder

img_ph = left[0].empty()
tbl_ph = right[0].empty()

while True:
  try:
    img_ph.image(FRAMEPATH, caption="Latest annotated frame", use_column_width=True)
  except Exception as e:
    img_ph.info(f"Could not load image. Waiting: {e}")
  
  rows = repo.fetch_pipes(limit=250)
  df = pd.DataFrame(rows, columns=["pipe_uid", "origin", "t_origin", "t_loadcell_enter", 
                                   "t_loadcell_exit", "avg_conf_full", "avg_conf_till_gate", 
                                   "frames_missing", "state", "last_seen_ts"])

  # Format timestamps first (while they may still be numeric/None)
  for c in ["t_origin", "t_loadcell_enter", "t_loadcell_exit", "last_seen_ts"]:
    if c in df.columns:
      df[c] = df[c].apply(fmt_ts)

  # Force text columns to plain Python objects to avoid Arrow LargeUtf8 in Streamlit
  text_cols = ["pipe_uid", "origin", "state", "t_origin", "t_loadcell_enter", "t_loadcell_exit", "last_seen_ts"]
  for c in text_cols:
    if c in df.columns:
      df[c] = df[c].map(lambda v: "" if v is None else str(v)).astype(object)

  tbl_ph.dataframe(df)
  time.sleep(refresh_ms / 1000.0)