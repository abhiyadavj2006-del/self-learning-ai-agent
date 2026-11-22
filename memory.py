"""Persistent memory for the self-learning agent -- Mem0-style.

This is the video's memory layer, rebuilt in stdlib + sqlite3 so it
runs with no key and no paid service:

* Mem0 API shape: ``add(messages, user_id=...)``,
  ``search(query, user_id=..., limit=3)``, ``get_all(...)``,
  ``get(memory_id)``, ``update(memory_id, data)``,
  ``delete(memory_id)`` and ``history(memory_id)`` -- the same calls
  Dave Ebbelaar's demo uses against Mem0 Cloud / the open-source
  ``Memory`` object (see README "Source").
* Two-phase pipeline (the video's core idea): the *extraction* phase
  pulls durable facts out of a conversation (Mem0 does this with an
  LLM + its FACT_RETRIEVAL_PROMPT; here a rule-based extractor runs
  by default and an LLM extractor takes over when an LLM client is
  passed in), then the *update* phase decides per fact whether to
  ADD, UPDATE, DELETE or do nothing (NONE) -- exactly Mem0's four
  operations. Contradictions are handled: "I don't like pizza
  anymore" removes pizza from ``likes`` and records it under
  ``dislikes`` (the video's pizza -> pasta demo).
* Multi-level memory: every fact is scoped to a ``user_id``,
  ``agent_id`` or ``session_id`` (Mem0's three levels: user =
  long-term, session = short-term, agent = agent long-term).
* History tracking: every ADD / UPDATE / DELETE is logged to the
  ``memory_events`` table, mirroring Mem0's ``history()`` (and its
  ``history_db_path`` config).
* Categories: facts are tagged like Mem0's fact-extraction prompt
  asks for -- preference, identity, plan, activity, health,
  professional, or plain fact.

Relevance search is token-overlap + recency, not vector embeddings:
real semantic search needs an embedder (Mem0's default is
``text-embedding-3-small``). See README "What was intentionally left
out".

The original key/value API (``set_fact`` / ``get_fact`` /
``list_facts`` / ``delete_fact`` / ``log`` / ``stats``) is kept
unchanged so existing code and tests keep working.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'user',
    category TEXT NOT NULL DEFAULT 'fact',
    user_id TEXT NOT NULL DEFAULT 'default',
    agent_id TEXT,
    session_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(key, user_id, scope)
);
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    user_input TEXT NOT NULL,
    response TEXT NOT NULL,
    tool_used TEXT
);
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER,
    key TEXT NOT NULL,
    event TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    user_id TEXT NOT NULL DEFAULT 'default',
    ts REAL NOT NULL
);
"""

# Mem0's open-source default. A fact row keeps its id across updates,
# so history() can follow one memory over time.
DEFAULT_USER_ID = "default"

# Categories from Mem0's fact-retrieval prompt: preferences, details,
# plans, activities, health information, professional details.
CATEGORIES = ("preference", "identity", "plan", "activity", "health",
              "professional", "fact")

_STOPWORDS = frozenset(
    "what is my the a an do you does did are was were i me tell recall "
    "about know did get give show list and or of for to in on".split())

# --------------------------------------------------------------------------
# fact extraction (phase 1 of the two-phase pipeline)
# --------------------------------------------------------------------------
# Split a message into clauses so "I don't like pizza anymore, I like
# pasta now" yields two facts instead of one garbled one.
_CLAUSE_SPLIT_RE = re.compile(r"[.;!]\s*|\s+but\s+|\s+and\s+", re.I)
_MY_IS_RE = re.compile(r"^my\s+(.+?)\s+is\s+(.+)$", re.I)
_REMEMBER_RE = re.compile(
    r"^(?:please\s+)?remember\s+(?:that\s+)?my\s+(.+?)\s+is\s+(.+)$", re.I)
_NEG_LIKE_RE = re.compile(
    r"^(?:i\s+)?(?:do\s+not|don't|dont|no\s+longer|not)\s+"
    r"(?:like|love|enjoy)\s+(.+?)(?:\s+anymore|\s+now)?$", re.I)
_NOT_FAN_RE = re.compile(
    r"^(?:i'?m\s+)?not\s+(?:a\s+)?(?:big\s+)?fan\s+of\s+(.+)$", re.I)
_LIKE_RE = re.compile(r"^i\s+(?:like|love|enjoy)\s+(.+)$", re.I)
_PLAN_RE = re.compile(r"^(?:i'?m|i\s+am)\s+planning\s+to\s+(.+)$", re.I)
_I_AM_RE = re.compile(r"^(?:i'?m|i\s+am)\s+(?:a\s+|an\s+)?(.+)$", re.I)

# Pronouns / filler that carry no fact on their own ("I don't like it
# anymore" -- the video shows even Mem0 gets this wrong sometimes).
_EMPTY_VALUES = {"it", "that", "this", "them", "those", "one", "anything"}


def _clean_value(value: str) -> str:
    value = value.strip().strip("\"'").rstrip(".!?,").strip()
    # "I like pasta now" / "don't like pizza anymore": the trailing
    # "now"/"anymore" is grammar, not part of the fact.
    value = re.sub(r"\s+(now|anymore)$", "", value, flags=re.I).strip()
    return " ".join(value.split())


def extract_facts_from_text(text: str) -> list[dict]:
    """Rule-based fact extractor (phase 1).

    Returns [{key, value, negated}] -- the same shape the LLM
    extractor in llm.py returns, so the two are interchangeable.
    Only user statements produce facts; questions and commands do not.
    """
    facts: list[dict] = []
    text = (text or "").strip()
    if not text or text.endswith("?"):
        return facts
    # "remember that my X is Y" is a statement about a fact, not a
    # bare "my X is Y" -- normalise it first.
    for clause in _CLAUSE_SPLIT_RE.split(text):
        clause = clause.strip().strip("\"'").rstrip(".!?").strip()
        if not clause:
            continue
        low = clause.lower()
        m = _REMEMBER_RE.match(clause) or _MY_IS_RE.match(clause)
        if m:
            key, value = normalise_key(m.group(1)), _clean_value(m.group(2))
            if key and value and value.lower() not in _EMPTY_VALUES:
                facts.append({"key": key, "value": value, "negated": False})
            continue
        m = _NEG_LIKE_RE.match(clause) or _NOT_FAN_RE.match(clause)
        if m:
            value = _clean_value(m.group(1))
            # Keep pronoun values ("it", "that") for negations: the
            # update phase tries to resolve them against the liked
            # things named in the same message (the video's pizza
            # demo); unresolvable ones are dropped there, never
            # guessed at here.
            if value:
                facts.append({"key": "likes", "value": value, "negated": True})
            continue
        m = _LIKE_RE.match(clause)
        if m:
            value = _clean_value(m.group(1))
            if value and value.lower() not in _EMPTY_VALUES:
                facts.append({"key": "likes", "value": value, "negated": False})
            continue
        m = _PLAN_RE.match(clause)
        if m:
            value = _clean_value(m.group(1))
            if value:
                facts.append({"key": "plan", "value": value, "negated": False})
            continue
        m = _I_AM_RE.match(clause)
        if m and len(clause.split()) <= 8:
            value = _clean_value(m.group(1))
            if value and value.lower() not in _EMPTY_VALUES:
                facts.append({"key": "identity", "value": value,
                              "negated": False})
    return facts


def classify_category(key: str, value: str = "") -> str:
    """Tag a fact the way Mem0's fact-retrieval prompt categorises."""
    key = key.lower()
    if key in {"name", "identity", "age", "location", "job"}:
        return "identity"
    if key in {"likes", "dislikes"} or "favourite" in key or "favorite" in key:
        return "preference"
    if key == "plan":
        return "plan"
    if any(w in key for w in ("health", "allerg", "diet", "sleep", "doctor")):
        return "health"
    if any(w in key for w in ("job", "work", "profession", "company", "boss")):
        return "professional"
    if any(w in key for w in ("hobby", "hobbies", "sport", "movie", "music")):
        return "activity"
    return "fact"


def normalise_key(raw: str) -> str:
    key = raw.strip().lower()
    if key.startswith("my "):
        key = key[3:]
    key = " ".join(key.split())
    return key


def _memory_text(key: str, value: str) -> str:
    """Human-readable memory string, like Mem0's stored memories."""
    if key == "likes":
        return f"likes {value}"
    if key == "dislikes":
        return f"does not like {value}"
    if key == "plan":
        return f"planning to {value}"
    if key == "identity":
        return value
    return f"{key} is {value}"


def _split_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


# Tiny synonym bridge: without embeddings, "favourite food" would
# never match a "likes pasta" memory. This maps the everyday words
# the video's own demo queries use ("what's my favourite food?") to
# the memory keys they mean. It is a keyword approximation of
# semantic search, not the real thing (see README).
_SYNONYMS = {
    "favourite": ["likes", "dislikes"],
    "favorite": ["likes", "dislikes"],
    "food": ["likes", "dislikes"],
    "eat": ["likes", "dislikes"],
    "drink": ["likes", "dislikes"],
    "hobby": ["likes"],
    "hobbies": ["likes"],
    "movie": ["likes", "plan"],
    "movies": ["likes", "plan"],
    "music": ["likes"],
    "sport": ["likes"],
}


def _token_variants(token: str) -> list[str]:
    return [token] + _SYNONYMS.get(token, [])


# --------------------------------------------------------------------------
class Memory:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate_facts_table()
        # Indexes are created after migration: an old-format DB has no
        # user_id/scope columns yet when SCHEMA first runs.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_user "
            "ON facts(user_id, scope)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_memory "
            "ON memory_events(memory_id)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- schema migration ----------------------------------------------
    def _migrate_facts_table(self) -> None:
        """Upgrade the starter schema (key PRIMARY KEY, no id/scope)
        to the Mem0-style schema, preserving existing rows."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(facts)")}
        if "id" in cols and "scope" in cols:
            return
        self.conn.execute(
            "CREATE TABLE facts_new ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " key TEXT NOT NULL, value TEXT NOT NULL,"
            " scope TEXT NOT NULL DEFAULT 'user',"
            " category TEXT NOT NULL DEFAULT 'fact',"
            " user_id TEXT NOT NULL DEFAULT 'default',"
            " agent_id TEXT, session_id TEXT,"
            " created_at REAL NOT NULL, updated_at REAL NOT NULL,"
            " UNIQUE(key, user_id, scope))")
        if "created_at" in cols:
            self.conn.execute(
                "INSERT INTO facts_new (key, value, created_at, updated_at)"
                " SELECT key, value, created_at, updated_at FROM facts")
        else:  # very old DBs without timestamps
            now = time.time()
            self.conn.execute(
                "INSERT INTO facts_new (key, value, created_at, updated_at)"
                " SELECT key, value, ?, ? FROM facts", (now, now))
        self.conn.execute("DROP TABLE facts")
        self.conn.execute("ALTER TABLE facts_new RENAME TO facts")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_user "
            "ON facts(user_id, scope)")

    # -- internal helpers ----------------------------------------------
    def _log_event(self, memory_id, key, event, old=None, new=None,
                   user_id: str = DEFAULT_USER_ID) -> None:
        self.conn.execute(
            "INSERT INTO memory_events (memory_id, key, event, old_value,"
            " new_value, user_id, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, key, event, old, new, user_id, time.time()))

    def _row(self, key, user_id, scope):
        return self.conn.execute(
            "SELECT * FROM facts WHERE key = ? AND user_id = ? AND scope = ?",
            (normalise_key(key), user_id, scope)).fetchone()

    def _insert_fact(self, key, value, scope, user_id, agent_id,
                     session_id) -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO facts (key, value, scope, category, user_id,"
            " agent_id, session_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, value, scope, classify_category(key, value), user_id,
             agent_id, session_id, now, now))
        return cur.lastrowid

    def _update_fact_value(self, row, value) -> None:
        self.conn.execute(
            "UPDATE facts SET value = ?, category = ?, updated_at = ?"
            " WHERE id = ?",
            (value, classify_category(row["key"], value),
             time.time(), row["id"]))

    # ---- original key/value API (unchanged behaviour) ------------------
    def set_fact(self, key: str, value: str, scope: str = "user",
                 category: str | None = None,
                 user_id: str = DEFAULT_USER_ID) -> bool:
        """Store/overwrite a fact. Returns True if it overwrote one."""
        key = normalise_key(key)
        value = " ".join(value.strip().split())
        if not key or not value:
            return False
        row = self._row(key, user_id, scope)
        if row is None:
            fid = self._insert_fact(key, value, scope, user_id, None, None)
            if category is not None:
                self.conn.execute("UPDATE facts SET category = ? WHERE id = ?",
                                  (category, fid))
            self._log_event(fid, key, "ADD", None, value, user_id)
        else:
            old = row["value"]
            self._update_fact_value(row, value)
            if category is not None:
                self.conn.execute("UPDATE facts SET category = ? WHERE id = ?",
                                  (category, row["id"]))
            self._log_event(row["id"], key, "UPDATE", old, value, user_id)
        self.conn.commit()
        return row is not None

    def get_fact(self, key: str, user_id: str = DEFAULT_USER_ID,
                 scope: str = "user") -> str | None:
        row = self._row(key, user_id, scope)
        return row["value"] if row else None

    def list_facts(self, scope: str | None = None,
                   user_id: str | None = None,
                   category: str | None = None) -> list[tuple[str, str]]:
        sql = "SELECT key, value FROM facts WHERE 1=1"
        args: list = []
        if scope:
            sql += " AND scope = ?"
            args.append(scope)
        if user_id:
            sql += " AND user_id = ?"
            args.append(user_id)
        if category:
            sql += " AND category = ?"
            args.append(category)
        sql += " ORDER BY key"
        rows = self.conn.execute(sql, args).fetchall()
        return [(r["key"], r["value"]) for r in rows]

    def delete_fact(self, key: str, user_id: str = DEFAULT_USER_ID,
                    scope: str = "user") -> bool:
        row = self._row(key, user_id, scope)
        if row is None:
            return False
        self.conn.execute("DELETE FROM facts WHERE id = ?", (row["id"],))
        self._log_event(row["id"], row["key"], "DELETE", row["value"], None,
                                row["user_id"])
        self.conn.commit()
        return True

    def find_facts_containing(self, word: str) -> list[tuple[str, str]]:
        """Facts whose key or value contains `word` (case-insensitive)."""
        like = f"%{word.lower()}%"
        rows = self.conn.execute(
            "SELECT key, value FROM facts "
            "WHERE lower(key) LIKE ? OR lower(value) LIKE ? ORDER BY key",
            (like, like)).fetchall()
        return [(r["key"], r["value"]) for r in rows]

    # ---- Mem0-style API ------------------------------------------------
    def add(self, messages, user_id: str | None = None,
            agent_id: str | None = None, session_id: str | None = None,
            metadata: dict | None = None, llm=None) -> dict:
        """Mem0-style ``add``: extract facts from messages, then
        ADD / UPDATE / DELETE / NONE each one (the update phase).

        ``messages`` is a string or a list of {"role", "content"}
        dicts (only ``user`` messages are mined for facts -- the
        assistant's replies are context, not facts). ``llm`` is an
        optional LLM client (see llm.py): when given, fact extraction
        uses the LLM + a Mem0-style extraction prompt; otherwise the
        rule-based extractor runs. Returns Mem0-shaped
        ``{"results": [{"id", "memory", "event"}]}``.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        user_id = user_id or DEFAULT_USER_ID
        scope = ("agent" if agent_id else
                 "session" if session_id else "user")
        user_texts = [m.get("content", "") for m in messages
                      if isinstance(m, dict) and m.get("role", "user") == "user"]
        facts: list[dict] = []
        if llm is not None:
            try:
                from llm import extract_facts_with_llm
                facts = extract_facts_with_llm(messages, llm) or []
            except Exception:
                facts = []
        if not facts:  # LLM unavailable / returned nothing: rules.
            for text in user_texts:
                facts.extend(extract_facts_from_text(text))
        if not facts and scope == "agent":
            # Agent-scope memories are usually instructions, not user
            # facts ("Return short responses when responding to
            # emails" -- Mem0's own quickstart example). A rule
            # extractor cannot parse imperatives, so store the
            # instruction verbatim instead of dropping it.
            for text in user_texts:
                text = " ".join(text.strip().split())
                if text:
                    facts.append({"key": "instruction",
                                  "value": text[:200], "negated": False})
        facts = self._resolve_pronouns(facts, user_texts, scope, user_id)
        results = []
        for f in facts:
            results.extend(self._apply_fact(
                f["key"], f["value"], f.get("negated", False),
                scope, user_id, agent_id, session_id))
        self.conn.commit()
        return {"results": results}

    def _resolve_pronouns(self, facts, user_texts, scope,
                          user_id) -> list[dict]:
        """Resolve "I don't like it anymore": "it" is the liked thing
        named in the same message. The video's demo relies on the LLM
        for this; the rule path only resolves it when the message
        names exactly one currently-liked thing, and drops the fact
        otherwise rather than guessing."""
        combined = " ".join(user_texts).lower()
        resolved: list[dict] = []
        for f in facts:
            if f.get("negated") and f["value"].lower() in _EMPTY_VALUES:
                row = self._row("likes", user_id, scope)
                candidates = [v for v in
                              (_split_list(row["value"]) if row else [])
                              if v.lower() in combined]
                if len(candidates) == 1:
                    resolved.append(dict(f, value=candidates[0]))
                continue
            resolved.append(f)
        return resolved

    def _apply_fact(self, key, value, negated, scope, user_id,
                    agent_id, session_id) -> list[dict]:
        """Update phase: decide ADD / UPDATE / DELETE / NONE for one
        extracted fact. Returns a list of Mem0-shaped result dicts
        (usually one; a contradicted like yields two: the DELETE of
        the old like and the ADD of the dislike)."""
        key = normalise_key(key)
        value = " ".join(value.strip().split())
        if not key or not value:
            return []
        if key == "likes":
            return self._apply_like(value, negated, scope, user_id,
                                    agent_id, session_id)
        row = self._row(key, user_id, scope)
        if negated:
            # "I don't like X anymore" for a non-likes key: if the
            # stored value mentions X, the memory is contradicted.
            if row and (value.lower() in row["value"].lower()
                        or row["value"].lower() in value.lower()):
                self.conn.execute("DELETE FROM facts WHERE id = ?",
                                  (row["id"],))
                self._log_event(row["id"], key, "DELETE",
                                row["value"], None, user_id)
                return [{"id": row["id"], "memory": _memory_text(key, value),
                         "event": "DELETE"}]
            return [{"id": None, "memory": _memory_text(key, value),
                     "event": "NONE"}]
        if row is None:
            fid = self._insert_fact(key, value, scope, user_id,
                                    agent_id, session_id)
            self._log_event(fid, key, "ADD", None, value, user_id)
            return [{"id": fid, "memory": _memory_text(key, value),
                     "event": "ADD"}]
        if row["value"] == value:
            return [{"id": row["id"], "memory": _memory_text(key, value),
                     "event": "NONE"}]
        old = row["value"]
        self._update_fact_value(row, value)
        self._log_event(row["id"], key, "UPDATE", old, value, user_id)
        return [{"id": row["id"], "memory": _memory_text(key, value),
                 "event": "UPDATE"}]

    def _apply_like(self, value, negated, scope, user_id,
                    agent_id, session_id) -> list[dict]:
        """Likes are a set, not a single value: "I like pizza" then
        "I like pasta" keeps both; "I don't like pizza anymore"
        removes pizza (the video's demo) and records a dislike."""
        row = self._row("likes", user_id, scope)
        current = _split_list(row["value"]) if row else []
        low = value.lower()
        results: list[dict] = []
        if negated:
            kept = [v for v in current if low not in v.lower()
                    and v.lower() not in low]
            removed = len(kept) != len(current)
            if row is not None and removed:
                if kept:
                    self._update_fact_value(row, ", ".join(kept))
                    self._log_event(row["id"], "likes", "UPDATE",
                                    row["value"], ", ".join(kept), user_id)
                    results.append(
                        {"id": row["id"],
                         "memory": _memory_text("likes", value),
                         "event": "UPDATE"})
                else:
                    self.conn.execute("DELETE FROM facts WHERE id = ?",
                                      (row["id"],))
                    self._log_event(row["id"], "likes", "DELETE",
                                    row["value"], None, user_id)
                    results.append(
                        {"id": row["id"],
                         "memory": _memory_text("likes", value),
                         "event": "DELETE"})
            # A stated dislike is itself a memory (Mem0 extracts
            # "not a big fan of thriller movies" as a fact).
            drow = self._row("dislikes", user_id, scope)
            if drow is None:
                fid = self._insert_fact("dislikes", value, scope, user_id,
                                        agent_id, session_id)
                self._log_event(fid, "dislikes", "ADD", None, value, user_id)
                results.append(
                    {"id": fid, "memory": _memory_text("dislikes", value),
                     "event": "ADD"})
                return results
            ditems = _split_list(drow["value"])
            if not any(low in v.lower() for v in ditems):
                new = ", ".join(ditems + [value])
                self._update_fact_value(drow, new)
                self._log_event(drow["id"], "dislikes", "UPDATE",
                                drow["value"], new, user_id)
                results.append(
                    {"id": drow["id"],
                     "memory": _memory_text("dislikes", value),
                     "event": "UPDATE"})
            else:
                results.append(
                    {"id": drow["id"],
                     "memory": _memory_text("dislikes", value),
                     "event": "NONE"})
            return results
        if any(low == v.lower() or low in v.lower() for v in current):
            return [{"id": row["id"], "memory": _memory_text("likes", value),
                     "event": "NONE"}]
        if row is None:
            fid = self._insert_fact("likes", value, scope, user_id,
                                    agent_id, session_id)
            self._log_event(fid, "likes", "ADD", None, value, user_id)
            return [{"id": fid, "memory": _memory_text("likes", value),
                     "event": "ADD"}]
        new = ", ".join(current + [value])
        self._update_fact_value(row, new)
        self._log_event(row["id"], "likes", "UPDATE", row["value"], new,
                        user_id)
        return [{"id": row["id"], "memory": _memory_text("likes", value),
                 "event": "UPDATE"}]

    def search(self, query: str, user_id: str | None = None,
               limit: int = 3, scope: str | None = None) -> list[dict]:
        """Mem0-style ``search``: relevance-ranked memories.

        Score = query-token coverage of the fact + a bonus for a
        token in the key + a bonus for the whole phrase + a small
        recency boost. Without embeddings this is keyword relevance,
        not semantic similarity (see README). Returns Mem0-shaped
        [{"id", "memory", "score", "metadata"}].
        """
        query = (query or "").strip().lower()
        if not query:
            return []
        tokens = [t for t in re.findall(r"[a-z0-9']+", query)
                  if t not in _STOPWORDS]
        if not tokens:
            tokens = re.findall(r"[a-z0-9']+", query)
        if not tokens:
            return []
        sql = "SELECT * FROM facts WHERE 1=1"
        args: list = []
        if user_id:
            sql += " AND user_id = ?"
            args.append(user_id)
        if scope:
            sql += " AND scope = ?"
            args.append(scope)
        rows = self.conn.execute(sql, args).fetchall()
        now = time.time()
        scored = []
        for r in rows:
            hay = f"{r['key']} {r['value']}".lower()
            hay_tokens = set(re.findall(r"[a-z0-9']+", hay))
            matched = 0
            for t in tokens:
                if any(v in hay_tokens or
                       any(v in h for h in hay_tokens)
                       for v in _token_variants(t)):
                    matched += 1
            if matched == 0:
                continue
            score = matched / len(tokens)
            if any(v in r["key"].lower()
                   for t in tokens for v in _token_variants(t)):
                score += 0.25
            if query in hay:
                score += 0.15
            age_days = max(0.0, (now - r["updated_at"]) / 86400.0)
            score += 0.05 * max(0.0, 1.0 - age_days / 30.0)
            scored.append({
                "id": r["id"],
                "memory": _memory_text(r["key"], r["value"]),
                "score": round(min(1.0, score), 3),
                "metadata": {"key": r["key"], "value": r["value"],
                             "category": r["category"], "scope": r["scope"],
                             "user_id": r["user_id"],
                             "updated_at": r["updated_at"]},
                "_updated_at": r["updated_at"],
            })
        scored.sort(key=lambda d: (d["score"], d["_updated_at"]),
                    reverse=True)
        for d in scored:
            d.pop("_updated_at", None)
        return scored[:max(1, limit)]

    def get_all(self, user_id: str | None = None,
                agent_id: str | None = None,
                session_id: str | None = None,
                limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM facts WHERE 1=1"
        args: list = []
        if user_id:
            sql += " AND user_id = ?"
            args.append(user_id)
        if agent_id:
            sql += " AND agent_id = ?"
            args.append(agent_id)
        if session_id:
            sql += " AND session_id = ?"
            args.append(session_id)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(max(1, limit))
        rows = self.conn.execute(sql, args).fetchall()
        return [{"id": r["id"], "memory": _memory_text(r["key"], r["value"]),
                 "key": r["key"], "value": r["value"],
                 "category": r["category"], "scope": r["scope"],
                 "user_id": r["user_id"], "agent_id": r["agent_id"],
                 "session_id": r["session_id"],
                 "updated_at": r["updated_at"]} for r in rows]

    def get(self, memory_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?",
                                (memory_id,)).fetchone()
        if row is None:
            return None
        return {"id": row["id"],
                "memory": _memory_text(row["key"], row["value"]),
                "key": row["key"], "value": row["value"],
                "category": row["category"], "scope": row["scope"],
                "user_id": row["user_id"], "updated_at": row["updated_at"]}

    def update(self, memory_id: int, data: str) -> bool:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?",
                                (memory_id,)).fetchone()
        if row is None:
            return False
        data = " ".join(str(data).strip().split())
        if not data:
            return False
        old = row["value"]
        self._update_fact_value(row, data)
        self._log_event(row["id"], row["key"], "UPDATE", old, data,
                            row["user_id"])
        self.conn.commit()
        return True

    def delete(self, memory_id: int) -> bool:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?",
                                (memory_id,)).fetchone()
        if row is None:
            return False
        self.conn.execute("DELETE FROM facts WHERE id = ?", (memory_id,))
        self._log_event(memory_id, row["key"], "DELETE", row["value"], None,
                            row["user_id"])
        self.conn.commit()
        return True

    def history(self, memory_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT event, old_value, new_value, ts FROM memory_events"
            " WHERE memory_id = ? ORDER BY ts, id",
            (memory_id,)).fetchall()
        return [{"event": r["event"], "old_value": r["old_value"],
                 "new_value": r["new_value"], "ts": r["ts"]} for r in rows]

    def history_for_key(self, key: str,
                        user_id: str = DEFAULT_USER_ID) -> list[dict]:
        """Full history of one key across memory ids: when a fact is
        deleted and re-added (e.g. likes pizza -> deleted -> likes
        pasta) the new row has a new id, but the story of the key
        spans both. Mem0's history() is per-id; this is the
        human-friendly view over it."""
        rows = self.conn.execute(
            "SELECT event, old_value, new_value, ts FROM memory_events"
            " WHERE key = ? AND user_id = ? ORDER BY ts, id",
            (normalise_key(key), user_id)).fetchall()
        return [{"event": r["event"], "old_value": r["old_value"],
                 "new_value": r["new_value"], "ts": r["ts"]} for r in rows]

    # ---- interaction log -------------------------------------------
    def log(self, user_input: str, response: str,
            tool_used: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO interactions (ts, user_input, response, tool_used)"
            " VALUES (?, ?, ?, ?)",
            (time.time(), user_input, response, tool_used))
        self.conn.commit()

    def stats(self) -> dict:
        n_facts = self.conn.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"]
        n_turns = self.conn.execute(
            "SELECT COUNT(*) c FROM interactions").fetchone()["c"]
        n_events = self.conn.execute(
            "SELECT COUNT(*) c FROM memory_events").fetchone()["c"]
        tools = self.conn.execute(
            "SELECT tool_used, COUNT(*) c FROM interactions "
            "WHERE tool_used IS NOT NULL GROUP BY tool_used ORDER BY c DESC"
        ).fetchall()
        return {
            "facts": n_facts,
            "interactions": n_turns,
            "history_events": n_events,
            "tools": {r["tool_used"]: r["c"] for r in tools},
        }
