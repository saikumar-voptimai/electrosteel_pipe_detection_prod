from __future__ import annotations
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import logging
from ui.formatting import fmt_ts

logger = logging.getLogger(__name__)

SCHEMA_SQL ="""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS pipes (
  pipe_uid TEXT PRIMARY KEY,
  tracker_id INTEGER,
  origin TEXT,
  pipe_checkpoint INTEGER DEFAULT 0,
  state TEXT,
  t_origin REAL,
  t_loadcell_enter REAL,
  t_loadcell_exit REAL,
  weight REAL,
  weight_quality TEXT,
  weight_samples INTEGER,
  avg_conf_full REAL,
  conf_count_full INTEGER,
  avg_conf_till_gate REAL,
  conf_count_till_gate INTEGER,
  frames_missing INTEGER,
  last_seen_ts REAL,
  reached_gate_zone INTEGER
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  event_type TEXT NOT NULL,
  pipe_uid TEXT,
  details TEXT
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS gate_cycles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  t_gate1_open REAL,
  t_gate1_close REAL,
  t_gate2_open REAL,
  t_gate2_close REAL,
  created_at REAL DEFAULT (strftime('%s','now'))
);

"""

@dataclass
class SqliteRepo:
  db_path: str

  def __post_init__(self) -> None:
    logger.info("Opening sqlite db: %s", self.db_path)
    self.conn = sqlite3.connect(self.db_path, 
                                timeout=30, 
                                check_same_thread=False)
    self.conn.executescript(SCHEMA_SQL)
    # Lightweight schema migration for existing DBs.
    # `CREATE TABLE IF NOT EXISTS` does not add columns to an existing table.
    self._ensure_columns(
      "pipes",
      {
        "weight": "REAL",
        "weight_quality": "TEXT",
        "weight_samples": "INTEGER",
        "pipe_checkpoint": "INTEGER DEFAULT 0",
      },
    )
    self.conn.commit()
    logger.debug("SQLite schema ensured")

  def _ensure_columns(self, table: str, cols: Dict[str, str]) -> None:
    cur = self.conn.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}  # row[1] = column name
    for name, typ in cols.items():
      if name in existing:
        continue
      logger.info("SQLite migrate | table=%s add_column=%s %s", table, name, typ)
      self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")

  def close(self) -> None:
    logger.info("Closing sqlite db")
    self.conn.close()

  def upsert_pipe(self, row: Dict[str, Any]) -> None:
    """
    Upsert a pipe record into the pipes table
    """
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    updates = ", ".join([f"{k}=excluded.{k}" for k in row.keys() if k != "pipe_uid"])

    sql = f"""
    INSERT INTO pipes ({cols}) VALUES ({placeholders})
    ON CONFLICT(pipe_uid) DO UPDATE SET {updates}
    """
    self.conn.execute(sql, tuple(row.values()))
    logger.debug("Upsert pipe | uid=%s | origin=%s | state=%s", row.get("pipe_uid"), row.get("origin"), row.get("state"))

  def delete_pipe(self, pipe_uid: str) -> None:
    self.conn.execute("DELETE FROM pipes WHERE pipe_uid=?", (pipe_uid,))
    logger.info("Deleted pipe | uid=%s", pipe_uid)
  
  def insert_event(self, event_type: str, pipe_uid: str | None, details: str = "") -> None:
    """
    Insert an event into the events table
    """ 
    self.conn.execute(
      "INSERT INTO events(ts,event_type,pipe_uid,details) VALUES(?,?,?,?)",
      (time.time(), event_type, pipe_uid, details)
    )
    logger.info("Event inserted | type=%s | pipe_uid=%s | details=%s", event_type, pipe_uid, details)
  
  def commit(self) -> None:
    logger.debug("DB commit")
    self.conn.commit()
  
  # UI queries
  def fetch_pipes(self, limit: int = 200) -> List[Tuple]:
    """
    fetch recent pipes for UI display
    """
    cursor = self.conn.execute(
      """
      SELECT pipe_uid, origin, pipe_checkpoint, t_origin, t_loadcell_enter, t_loadcell_exit,
              weight, weight_quality, weight_samples,
             avg_conf_full, avg_conf_till_gate, frames_missing, state, last_seen_ts
      FROM pipes
      ORDER BY COALESCE(t_origin, 0) DESC
      LIMIT ?
      """,
      (limit,)
    )
    return cursor.fetchall()
  
  def metric_counts(self, seconds: int) -> int:
    """
    Metric for counting pipes from caster origin in last `seconds`
    """
    cutoff_ts = time.time() - seconds
    cursor = self.conn.execute(
      """
      SELECT COUNT(*) FROM pipes WHERE origin='caster' AND t_origin IS NOT NULL AND t_origin >= ?
      """,
      (cutoff_ts,),
    )
    return int(cursor.fetchone()[0])
  
  def metric_avg_conf(self, seconds: int) -> float:
    """
    Metric for average detection-confidence of pipes from caster origin in last `seconds`
    """
    cutoff_ts = time.time() - seconds
    cursor = self.conn.execute(
      """
      SELECT AVG(avg_conf_full) FROM pipes 
      WHERE origin ='caster' AND t_origin IS NOT NULL AND t_origin >= ? AND avg_conf_full IS NOT NULL
      """,
      (cutoff_ts,),
    )
    v = cursor.fetchone()[0]
    return float(v) if v is not None else 0.0
  
  # Settings for live switching
  def set_setting(self, key: str, value: str) -> None:
    """
    Set a setting key-value pair
    """
    self.conn.execute(
      """
      INSERT INTO settings(key,value,updated_at) VALUES(?,?,?)
      ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
      """,
      (key, value, time.time())
    )
    self.conn.commit()
  
  def get_setting(self, key: str, default: str) -> str:
    """
    Get a setting value by key
    """
    cur = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default
  
  def gate_open(self, gate, ts):

    col = "t_gate1_open" if gate == "gate1" else "t_gate2_open"

    sql = f"""
    UPDATE gate_cycles
    SET {col} = ?
    WHERE id = (
        SELECT id FROM gate_cycles
        WHERE t_gate1_close IS NULL
          OR t_gate2_close IS NULL
        ORDER BY id DESC
        LIMIT 1
    )
    """

    cur = self.conn.execute(sql, (ts,))

    if cur.rowcount == 0:
        sql = f"INSERT INTO gate_cycles ({col}) VALUES (?)"
        self.conn.execute(sql, (ts,))

  def gate_close(self, gate, ts):

    col = "t_gate1_close" if gate == "gate1" else "t_gate2_close"

    sql = f"""
    UPDATE gate_cycles
    SET {col} = ?
    WHERE id = (
        SELECT id FROM gate_cycles
        WHERE {col} IS NULL
        ORDER BY id DESC
        LIMIT 1
    )
    """

    self.conn.execute(sql, (ts,))



     


