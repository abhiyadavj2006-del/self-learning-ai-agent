#!/usr/bin/env python3
"""Self-learning AI agent -- rule-based reasoning, SQLite memory.

No LLM, no API key, no network: stdlib + sqlite3 only.

Learning loop (the "self-learning" part, deliberately small and real):
  1. Observe: every turn is logged to the interactions table.
  2. Learn:  "remember that my favourite colour is blue" / "my name is
     Ada" / "I like hiking" stores a (key, value) fact in SQLite.
  3. Recall: "what is my favourite colour?" reads it back, even after
     the process exits and restarts -- the DB file is the memory.
  4. Correct: telling it the same key again overwrites the old value,
     so a correction ("no, my favourite colour is green") is learned.

Tools (see tools.py): calculator (safe AST maths) and note search
(full-text search over notes/*.txt|*.md).

Usage:
  python3 agent.py                  interactive REPL
  python3 agent.py --ask "what is 12 * 8?"     single question, then exit
  python3 agent.py --db /tmp/test.db --ask "..."  use a different DB
  python3 agent.py --notes notes     notes dir (default: notes/ next to this file)

Exit the REPL with: exit / quit / Ctrl-D.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from memory import Memory, normalise_key  # noqa: E402
from tools import calculate, list_notes, search_notes  # noqa: E402

HELP = """I can:
  - Remember facts:  remember that my favourite colour is blue
                      my name is Ada
                      remember my dog's name is Rex
  - Recall them:     what is my favourite colour? / what do you know about me?
                     list facts
  - Forget them:     forget my favourite colour
  - Calculate:       calculate 12 * (3 + 4)   /   what is 45 + 23?
  - Search notes:    search notes for python   /   find in notes hiking
  - Show stats:      stats
  - Quit:            exit / quit
"""

# Patterns are tried in order in respond(); first match wins.
REMEMBER_RE = re.compile(
    r"^(?:please\s+)?remember\s+(?:that\s+)?(.+?)\s+is\s+(.+)$", re.I)
REMEMBER_LIKE_RE = re.compile(
    r"^(?:i\s+like|i\s+love|i\s+enjoy)\s+(.+)$", re.I)
MY_IS_RE = re.compile(r"^my\s+(.+?)\s+is\s+(.+)$", re.I)
CORRECTION_RE = re.compile(r"^no,?\s+my\s+(.+?)\s+is\s+(.+)$", re.I)
RECALL_RE = re.compile(
    r"^(?:what(?:'s| is)|do you remember|tell me|recall|what did i tell you about)"
    r"\s+(?:my\s+)?(.+?)\??$", re.I)
FORGET_RE = re.compile(r"^forget\s+(?:my\s+)?(.+?)\s*$", re.I)
CALC_RE = re.compile(r"^(?:calculate|calc|compute|work out)\s+(.+)$", re.I)
CALC_Q_RE = re.compile(r"^what is\s+([0-9][0-9\s+\-*/%().]+)\??$", re.I)
SEARCH_RE = re.compile(
    r"^(?:search(?: my)? notes? for|find in (?:my )?notes?|search for)\s+(.+)$", re.I)


class Agent:
    def __init__(self, db_path: str | Path, notes_dir: str | Path):
        self.memory = Memory(db_path)
        self.notes_dir = Path(notes_dir)

    def close(self) -> None:
        self.memory.close()

    # -- helpers ------------------------------------------------------
    def _learn(self, key: str, value: str) -> str:
        key = normalise_key(key)  # "my favourite colour" -> "favourite colour"
        overwritten = self.memory.set_fact(key, value)
        verb = "Updated" if overwritten else "Got it"
        return f"{verb} -- I'll remember that your {key} is {value.strip()}."

    def _recall(self, key: str) -> str:
        value = self.memory.get_fact(key)
        if value is not None:
            return f"Your {key.strip().lower()} is {value}."
        close = self.memory.find_facts_containing(key.strip().lower())
        if close:
            lines = "\n".join(f"  - {k}: {v}" for k, v in close)
            return f"I don't have an exact fact for '{key}', but I found:\n{lines}"
        return (f"I don't know your {key.strip().lower()} yet. "
                "Tell me with: remember that my "
                f"{key.strip().lower()} is ...")

    # -- main dispatch -------------------------------------------------
    def respond(self, user_input: str) -> tuple[str, str | None]:
        """Return (reply, tool_used_or_None). Logs the turn."""
        text = user_input.strip()
        low = text.lower()
        # canonical form for exact-command matches: ignore a trailing "?"
        canon = low.rstrip("?").strip()
        tool: str | None = None

        if not text:
            reply = "Say something -- try 'help'."
        elif low in {"exit", "quit", "bye"}:
            reply = "__EXIT__"
        elif low in {"help", "?"}:
            reply = HELP
        elif canon in {"list facts", "show facts", "show memory",
                       "what do you know about me", "what do you know"}:
            facts = self.memory.list_facts()
            reply = ("I don't know anything about you yet. Tell me a fact "
                     "with 'remember that my ... is ...'."
                     if not facts else
                     "Here's what I remember about you:\n" +
                     "\n".join(f"  - {k}: {v}" for k, v in facts))
        elif low == "stats":
            s = self.memory.stats()
            tools = ", ".join(f"{k}: {v}" for k, v in s["tools"].items()) or "none yet"
            reply = (f"Memory: {s['facts']} fact(s), {s['interactions']} "
                     f"interaction(s) logged. Tools used: {tools}.")
        elif low in {"list notes", "show notes"}:
            notes = list_notes(self.notes_dir)
            reply = ("No notes found in " + str(self.notes_dir) if not notes else
                     "Notes I can search:\n" + "\n".join(f"  - {n}" for n in notes))
        elif (m := CORRECTION_RE.match(text)) or (m := REMEMBER_RE.match(text)):
            reply = self._learn(m.group(1), m.group(2))
        elif (m := MY_IS_RE.match(text)) and len(text.split()) >= 4:
            # "my favourite colour is blue" -- a bare statement, learn it.
            reply = self._learn(m.group(1), m.group(2))
        elif m := REMEMBER_LIKE_RE.match(text):
            reply = self._learn("likes", m.group(1))
        elif m := FORGET_RE.match(text):
            key = m.group(1)
            reply = (f"Forgot your {key}." if self.memory.delete_fact(key)
                     else f"I had nothing stored for '{key}'.")
        elif m := CALC_RE.match(text):
            tool = "calculator"
            try:
                reply = f"{m.group(1).strip()} = {calculate(m.group(1))}"
            except ValueError as e:
                reply = f"Calculator error: {e}"
        elif m := CALC_Q_RE.match(text):
            tool = "calculator"
            try:
                reply = f"{m.group(1).strip()} = {calculate(m.group(1))}"
            except ValueError as e:
                reply = f"Calculator error: {e}"
        elif m := SEARCH_RE.match(text):
            tool = "note_search"
            hits = search_notes(m.group(1), self.notes_dir)
            if not hits:
                reply = f"No notes mention '{m.group(1)}'."
            else:
                shown = hits[:5]
                lines = "\n".join(
                    f"  - {h['file']}:{h['line']}: {h['text']}" for h in shown)
                extra = f"\n  ... and {len(hits) - 5} more match(es)." \
                    if len(hits) > 5 else ""
                reply = f"Found {len(hits)} match(es) for '{m.group(1)}':\n{lines}{extra}"
        elif m := RECALL_RE.match(text):
            key = m.group(1).strip()
            # "what is 12+3?" was handled above; "what is my X?" lands here.
            # Guard: a bare "what is <word>" with no stored fact still goes
            # through recall and gets the honest "I don't know yet" reply.
            reply = self._recall(key)
        else:
            # Last resort: does the input mention a stored key verbatim?
            # e.g. "favourite colour?" -> recall it.
            for key, value in self.memory.list_facts():
                if key in low:
                    reply = f"Your {key} is {value}."
                    break
            else:
                reply = ("I don't know how to answer that yet. I can remember "
                         "facts ('remember that my ... is ...'), recall them "
                         "('what is my ...?'), calculate ('calculate 12 * 8'), "
                         "or search notes ('search notes for ...'). "
                         "Try 'help'.")

        if reply != "__EXIT__":
            self.memory.log(user_input, reply, tool)
        return reply, tool


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Self-learning AI agent (stdlib only)")
    p.add_argument("--ask", help="ask one question, print the reply, exit")
    p.add_argument("--db", default=str(BASE / "agent_memory.db"),
                   help="SQLite DB path (default: agent_memory.db)")
    p.add_argument("--notes", default=str(BASE / "notes"),
                   help="notes directory for the search tool")
    args = p.parse_args(argv)

    agent = Agent(args.db, args.notes)
    try:
        if args.ask is not None:
            reply, _ = agent.respond(args.ask)
            print(reply if reply != "__EXIT__" else "Goodbye.")
            return 0
        print("Luna-agent: a tiny self-learning agent. Type 'help', or 'exit' to quit.")
        print(f"Memory: {args.db}  |  Notes: {args.notes}")
        while True:
            try:
                line = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye -- everything you taught me is saved.")
                break
            reply, _ = agent.respond(line)
            if reply == "__EXIT__":
                print("Goodbye -- everything you taught me is saved.")
                break
            print(f"agent> {reply}")
        return 0
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
