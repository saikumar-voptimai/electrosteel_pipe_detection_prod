from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db.repo import SqliteRepo
from logic.gate_fsm import GateFSM


class _GateSource:
    def __init__(self) -> None:
        self.position = "closed"

    def get_position(self, gate_name, frame=None, dets=None):
        return self.position, {}


class _RecordingPLC:
    def __init__(self) -> None:
        self.pulses: list[tuple[str, int]] = []

    def pulse(self, tag: str, pulse_ms: int) -> None:
        self.pulses.append((tag, pulse_ms))


class GateOpeningOnlyTests(unittest.TestCase):
    def test_closed_state_rearms_without_emitting_close_events(self) -> None:
        source = _GateSource()
        plc = _RecordingPLC()
        fsm = GateFSM(
            source=source,
            plc=plc,
            pulse_ms=250,
            stable_frames=2,
            gate_tags={"gate1": "gate1_open", "gate2": "gate2_open"},
        )

        self.assertEqual(fsm.update()[0], [])
        self.assertEqual(fsm.update()[0], [])

        source.position = "open"
        self.assertEqual(fsm.update()[0], [])
        first_openings = fsm.update()[0]
        self.assertEqual([event.gate_name for event in first_openings], ["gate1", "gate2"])

        source.position = "closed"
        self.assertEqual(fsm.update()[0], [])
        self.assertEqual(fsm.update()[0], [])

        source.position = "open"
        self.assertEqual(fsm.update()[0], [])
        second_openings = fsm.update()[0]
        self.assertEqual([event.gate_name for event in second_openings], ["gate1", "gate2"])

    def test_new_database_schema_and_writes_are_opening_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = SqliteRepo(str(Path(tmp) / "pipes.db"))
            try:
                columns = {
                    row[1] for row in repo.conn.execute("PRAGMA table_info(gate_cycles)").fetchall()
                }
                self.assertIn("t_gate1_open", columns)
                self.assertIn("t_gate2_open", columns)
                self.assertNotIn("t_gate1_close", columns)
                self.assertNotIn("t_gate2_close", columns)

                repo.gate_open("gate1", 100.0)
                repo.gate_open("gate2", 200.0)
                repo.commit()

                self.assertEqual(
                    repo.conn.execute(
                        "SELECT gate_name, t_open FROM gate_openings ORDER BY id"
                    ).fetchall(),
                    [("gate1", 100.0), ("gate2", 200.0)],
                )
                self.assertEqual(
                    repo.conn.execute(
                        "SELECT t_gate1_open, t_gate2_open FROM gate_cycles ORDER BY id"
                    ).fetchall(),
                    [(100.0, 200.0)],
                )
            finally:
                repo.close()

    def test_existing_database_keeps_historical_close_columns_but_does_not_write_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "legacy.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE gate_cycles (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  t_gate1_open REAL,
                  t_gate1_close REAL,
                  t_gate2_open REAL,
                  t_gate2_close REAL,
                  created_at REAL
                )
                """
            )
            conn.execute(
                "INSERT INTO gate_cycles(t_gate1_open,t_gate1_close) VALUES(?,?)",
                (10.0, 20.0),
            )
            conn.commit()
            conn.close()

            repo = SqliteRepo(str(db_path))
            try:
                repo.gate_open("gate1", 30.0)
                repo.commit()
                rows = repo.conn.execute(
                    "SELECT t_gate1_open,t_gate1_close FROM gate_cycles ORDER BY id"
                ).fetchall()
                self.assertEqual(rows, [(10.0, 20.0), (30.0, None)])
            finally:
                repo.close()


if __name__ == "__main__":
    unittest.main()
