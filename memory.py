"""Persistent memory for the self-learning agent.

Stdlib only: sqlite3. One table for facts the agent has learned,
one table for the interaction log (the raw material for learning:
what was asked, what tool was used, what was answered).

A "fact" is a (key, value) pair, e.g. key="favourite colour",
value="blue". Keys are normalised to lowercase, stripped of a
leading "my ". Setting the same key again overwrites it -- that
is how the agent learns from a correction ("no, my favourite
colour is green").
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    user_input TEXT NOT NULL,
    response TEXT NOT NULL,
    tool_used TEXT
);
"""


def normalise_key(raw: str) -> str:
    key = raw.strip().lower()
    if key.startswith("my "):
        key = key[3:]
    # collapse internal whitespace
    key = " ".join(key.split())
    return key


class Memory:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- facts ------------------------------------------------------
    def set_fact(self, key: str, value: str) -> bool:
        """Store/overwrite a fact. Returns True if it overwrote one."""
        key = normalise_key(key)
        value = " ".join(value.strip().split())
        if not key or not value:
            return False
        existed = self.get_fact(key) is not None
        now = time.time()
        self.conn.execute(
            """INSERT INTO facts (key, value, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE
               SET value = excluded.value, updated_at = excluded.updated_at""",
            (key, value, now, now),
        )
        self.conn.commit()
        return existed

    def get_fact(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM facts WHERE key = ?", (normalise_key(key),)
        ).fetchone()
        return row["value"] if row else None

    def list_facts(self) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT key, value FROM facts ORDER BY key"
        ).fetchall()
        return [(r["key"], r["value"]) for r in rows]

    def delete_fact(self, key: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM facts WHERE key = ?", (normalise_key(key),)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def find_facts_containing(self, word: str) -> list[tuple[str, str]]:
        """Facts whose key or value contains `word` (case-insensitive)."""
        like = f"%{word.lower()}%"
        rows = self.conn.execute(
            "SELECT key, value FROM facts "
            "WHERE lower(key) LIKE ? OR lower(value) LIKE ? ORDER BY key",
            (like, like),
        ).fetchall()
        return [(r["key"], r["value"]) for r in rows]

    # ---- interaction log -------------------------------------------
    def log(self, user_input: str, response: str, tool_used: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO interactions (ts, user_input, response, tool_used)"
            " VALUES (?, ?, ?, ?)",
            (time.time(), user_input, response, tool_used),
        )
        self.conn.commit()

    def stats(self) -> dict:
        n_facts = self.conn.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"]
        n_turns = self.conn.execute(
            "SELECT COUNT(*) c FROM interactions"
        ).fetchone()["c"]
        tools = self.conn.execute(
            "SELECT tool_used, COUNT(*) c FROM interactions "
            "WHERE tool_used IS NOT NULL GROUP BY tool_used ORDER BY c DESC"
        ).fetchall()
        return {
            "facts": n_facts,
            "interactions": n_turns,
            "tools": {r["tool_used"]: r["c"] for r in tools},
        }
