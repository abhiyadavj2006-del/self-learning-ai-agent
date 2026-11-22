#!/usr/bin/env python3
"""Self-learning AI agent -- Mem0-style memory + optional LLM, Python.

The learning loop follows Dave Ebbelaar's "How to Build Self-Learning
AI Agents in Python" (Mem0): every turn, retrieve the relevant
memories (``memory.search(..., limit=3)``), answer with them in
context, then write the conversation back to memory, where the
two-phase pipeline extracts facts and decides ADD / UPDATE / DELETE /
NONE per fact (see memory.py).

Modes (no key needed to run):
  * Rule-based (default): stdlib + sqlite3 only. The agent learns
    from "remember that my X is Y" / "I like X" patterns, recalls
    with relevance-ranked search, and corrects itself.
  * + LLM (optional): set OPENAI_API_KEY (or point OPENAI_BASE_URL at
    a local Ollama server) and the agent answers open questions with
    the LLM, memories injected into the system prompt -- the video's
    demo loop, see Agent.chat_with_memories().
  * + real Mem0 (optional): set MEM0_API_KEY (cloud) or install
    mem0ai (open source, Qdrant via docker-compose.yml) and every
    fact is mirrored into Mem0 as well (see mem0_backend.py).

Tools (see tools.py): calculator (safe AST maths) and note search
(full-text search over notes/*.txt|*.md).

Usage:
  python3 agent.py                  interactive REPL
  python3 agent.py --ask "what is 12 * 8?"     single question, then exit
  python3 agent.py --db /tmp/test.db --ask "..."  use a different DB
  python3 agent.py --notes notes     notes dir (default: notes/ next to this file)
  python3 agent.py --user alice --no-llm        per-user memory, rule-based

Exit the REPL with: exit / quit / Ctrl-D.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from memory import DEFAULT_USER_ID, Memory, normalise_key  # noqa: E402
from tools import calculate, list_notes, search_notes  # noqa: E402
from llm import (MEMORY_ANSWER_SYSTEM_PROMPT,  # noqa: E402
                 get_llm_client)
from mem0_backend import get_mem0_backend  # noqa: E402

HELP = """I can:
  - Remember facts:  remember that my favourite colour is blue
                      my name is Ada
                      remember my dog's name is Rex
                      I like pizza / I don't like pizza anymore
  - Recall them:     what is my favourite colour? / what do you know about me?
                     list facts / memories
  - Search memory:   search memory favourite food   (relevance-ranked)
  - Memory history:  history favourite colour       (ADD/UPDATE/DELETE log)
  - Forget them:     forget my favourite colour
  - Calculate:       calculate 12 * (3 + 4)   /   what is 45 + 23?
  - Search notes:    search notes for python   /   find in notes hiking
  - Show stats/mode: stats / mode
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


SEARCH_MEMORY_RE = re.compile(r"^search memory\s+(.+)$", re.I)
HISTORY_RE = re.compile(r"^history\s+(?:of\s+)?(?:my\s+)?(.+?)\s*$", re.I)


class Agent:
    def __init__(self, db_path: str | Path, notes_dir: str | Path,
                 user_id: str = DEFAULT_USER_ID,
                 llm="auto", mem0="auto"):
        """llm/mem0: an object to use, None to disable, "auto" (the
        default) to detect from env vars. Auto-detection never needs
        a key: with nothing configured both stay off and the agent is
        rule-based."""
        self.memory = Memory(db_path)
        self.notes_dir = Path(notes_dir)
        self.user_id = user_id
        self.llm = get_llm_client() if llm == "auto" else llm
        self.mem0 = get_mem0_backend() if mem0 == "auto" else mem0

    def close(self) -> None:
        self.memory.close()

    @property
    def mode(self) -> str:
        llm = f"LLM: {self.llm.model} ({self.llm.provider})" \
            if self.llm else "LLM: off (rule-based)"
        mem0 = f"Mem0: {self.mem0.describe()}" \
            if self.mem0 else "Mem0: off"
        return (f"Memory: SQLite (Mem0-style) | {llm} | {mem0} | "
                f"user: {self.user_id}")

    # -- helpers ------------------------------------------------------
    def _learn(self, key: str, value: str) -> str:
        key = normalise_key(key)  # "my favourite colour" -> "favourite colour"
        overwritten = self.memory.set_fact(key, value, user_id=self.user_id)
        self._mirror_mem0(f"My {key} is {value.strip()}.")
        verb = "Updated" if overwritten else "Got it"
        return f"{verb} -- I'll remember that your {key} is {value.strip()}."

    def _mirror_mem0(self, text: str) -> None:
        """Best-effort mirror into real Mem0 when configured; never
        breaks the local flow."""
        if self.mem0 is not None and getattr(self.mem0, "available", False):
            try:
                self.mem0.add([{"role": "user", "content": text}],
                              user_id=self.user_id)
            except Exception:
                pass

    def _recall(self, key: str) -> str:
        value = self.memory.get_fact(key, user_id=self.user_id)
        if value is not None:
            return f"Your {key.strip().lower()} is {value}."
        # Relevance-ranked recall (Mem0-style search), not just a
        # substring fallback.
        close = self.memory.search(key, user_id=self.user_id, limit=3)
        if close:
            lines = "\n".join(
                f"  - {c['memory']} (score {c['score']})" for c in close)
            return (f"I don't have an exact fact for '{key}', but the "
                    f"closest memories are:\n{lines}")
        return (f"I don't know your {key.strip().lower()} yet. "
                "Tell me with: remember that my "
                f"{key.strip().lower()} is ...")

    def chat_with_memories(self, user_input: str) -> str | None:
        """The video's demo loop, step by step:

        1. Retrieve: ``memory.search(query, user_id, limit=3)``.
        2. Answer: LLM with the memories injected into the system
           prompt (Mem0's MEMORY_ANSWER_PROMPT idea).
        3. Write back: ``memory.add(messages)`` -- extract facts from
           the turn and ADD / UPDATE / DELETE / NONE each one.
        Returns None when no LLM is available (caller falls back).
        """
        if self.llm is None:
            return None
        relevant = self.memory.search(user_input, user_id=self.user_id,
                                      limit=3)
        if relevant:
            memories_text = "\n".join(f"- {r['memory']}"
                                      for r in relevant)
        else:
            memories_text = "(no relevant memories yet)"
        system = (MEMORY_ANSWER_SYSTEM_PROMPT +
                  f"\n\nRelevant memories about the user:\n{memories_text}")
        reply = self.llm.generate(
            [{"role": "user", "content": user_input}], system=system)
        if reply is None:
            return None
        # Write the turn back to memory (extraction + update phase).
        self.memory.add(
            [{"role": "user", "content": user_input},
             {"role": "assistant", "content": reply}],
            user_id=self.user_id, llm=self.llm)
        self._mirror_mem0(user_input)
        return reply


    # -- main dispatch -------------------------------------------------
    def respond(self, user_input: str) -> tuple[str, str | None]:
        """Return (reply, tool_used_or_None). Logs the turn."""
        text = user_input.strip()
        low = text.lower()
        # canonical form for exact-command matches: ignore a trailing "?"
        canon = low.rstrip("?").strip()
        tool: str | None = None
        learned = False  # a rule already stored this turn's fact
        llm_answered = False  # the LLM demo loop already wrote memory
        fell_back = False  # no rule/tool matched; honest fallback reply

        if not text:
            reply = "Say something -- try 'help'."
        elif low in {"exit", "quit", "bye"}:
            reply = "__EXIT__"
        elif low in {"help", "?"}:
            reply = HELP
        elif low == "mode":
            reply = self.mode
        elif canon in {"list facts", "show facts", "show memory",
                       "what do you know about me", "what do you know",
                       "list memories", "show memories", "memories"}:
            facts = self.memory.get_all(user_id=self.user_id)
            reply = ("I don't know anything about you yet. Tell me a fact "
                     "with 'remember that my ... is ...'."
                     if not facts else
                     "Here's what I remember about you:\n" +
                     "\n".join(f"  - {f['memory']} [{f['category']}, "
                               f"{f['scope']}]" for f in facts))
        elif m := SEARCH_MEMORY_RE.match(text):
            hits = self.memory.search(m.group(1), user_id=self.user_id,
                                      limit=5)
            if not hits:
                reply = f"No memories match '{m.group(1)}'."
            else:
                lines = "\n".join(
                    f"  - {h['memory']} (score {h['score']})" for h in hits)
                reply = (f"Most relevant memories for "
                         f"'{m.group(1)}':\n{lines}")
        elif m := HISTORY_RE.match(text):
            key = normalise_key(m.group(1))
            events = self.memory.history_for_key(key,
                                                 user_id=self.user_id)
            if not events:
                reply = f"I have no memory history for '{key}'."
            else:
                lines = "\n".join(
                    f"  - {e['event']}: "
                    f"{e['old_value'] or '—'} -> "
                    f"{e['new_value'] or '—'}" for e in events)
                reply = f"History of '{key}':\n{lines}"
        elif low == "stats":
            s = self.memory.stats()
            tools = ", ".join(f"{k}: {v}" for k, v in s["tools"].items()) or "none yet"
            reply = (f"Memory: {s['facts']} fact(s), {s['interactions']} "
                     f"interaction(s) logged, {s['history_events']} "
                     f"memory event(s). Tools used: {tools}.\n{self.mode}")
        elif low in {"list notes", "show notes"}:
            notes = list_notes(self.notes_dir)
            reply = ("No notes found in " + str(self.notes_dir) if not notes else
                     "Notes I can search:\n" + "\n".join(f"  - {n}" for n in notes))
        elif (m := CORRECTION_RE.match(text)) or (m := REMEMBER_RE.match(text)):
            reply = self._learn(m.group(1), m.group(2))
            learned = True
        elif (m := MY_IS_RE.match(text)) and len(text.split()) >= 4:
            # "my favourite colour is blue" -- a bare statement, learn it.
            reply = self._learn(m.group(1), m.group(2))
            learned = True
        elif m := REMEMBER_LIKE_RE.match(text):
            reply = self._learn("likes", m.group(1))
            learned = True
        elif m := FORGET_RE.match(text):
            key = m.group(1)
            reply = (f"Forgot your {key}." if self.memory.delete_fact(
                         key, user_id=self.user_id)
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
            for key, value in self.memory.list_facts(user_id=self.user_id):
                if key in low:
                    reply = f"Your {key} is {value}."
                    break
            else:
                # The video's demo loop: with an LLM configured, answer
                # from retrieved memories; without one, say so honestly.
                reply = self.chat_with_memories(text)
                if reply is None:
                    fell_back = True
                    reply = ("I don't know how to answer that yet. I can remember "
                             "facts ('remember that my ... is ...'), recall them "
                             "('what is my ...?'), calculate ('calculate 12 * 8'), "
                             "or search notes ('search notes for ...'). "
                             "Try 'help'. (Tip: set OPENAI_API_KEY to let me "
                             "answer open questions with an LLM.)")
                else:
                    llm_answered = True

        # Self-learning write-back (the video's update phase): if this
        # turn was a plain statement the rules didn't already learn
        # from, mine it for durable facts -- ADD / UPDATE / DELETE /
        # NONE, exactly like memory.add() after every chat turn in
        # the demo. Questions, commands and tool calls are skipped.
        if (reply != "__EXIT__" and not learned and not llm_answered
                and tool is None and not text.endswith("?")
                and low not in {"help", "?", "stats", "mode"}
                and not canon.startswith(("list facts", "show facts",
                                          "show memory", "list memories",
                                          "show memories", "memories",
                                          "list notes", "show notes",
                                          "search memory", "history"))):
            res = self.memory.add([{"role": "user", "content": text}],
                                  user_id=self.user_id, llm=self.llm)
            self._mirror_mem0(text)
            if fell_back:
                # The turn taught the agent something even though no
                # rule could answer it -- say what was learned instead
                # of the generic "I don't know".
                noted = [r for r in res.get("results", [])
                         if r["event"] in ("ADD", "UPDATE", "DELETE")]
                if noted:
                    reply = ("Noted — I've updated what I remember:\n" +
                             "\n".join(
                                 f"  - {r['event'].title()}: {r['memory']}"
                                 for r in noted))

        if reply != "__EXIT__":
            self.memory.log(user_input, reply, tool)
        return reply, tool


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Self-learning AI agent (Mem0-style memory; "
                    "LLM + real Mem0 optional)")
    p.add_argument("--ask", help="ask one question, print the reply, exit")
    p.add_argument("--db", default=str(BASE / "agent_memory.db"),
                   help="SQLite DB path (default: agent_memory.db)")
    p.add_argument("--notes", default=str(BASE / "notes"),
                   help="notes directory for the search tool")
    p.add_argument("--user", default=DEFAULT_USER_ID,
                   help="user_id for memory scoping (default: default)")
    p.add_argument("--model", default=None,
                   help="LLM model (default: $LLM_MODEL or gpt-4o-mini)")
    p.add_argument("--base-url", default=None,
                   help="LLM base URL, e.g. http://localhost:11434/v1 "
                        "for Ollama (default: $OPENAI_BASE_URL)")
    p.add_argument("--no-llm", action="store_true",
                   help="force rule-based mode even if a key is set")
    p.add_argument("--no-mem0", action="store_true",
                   help="disable the real-Mem0 mirror even if configured")
    args = p.parse_args(argv)

    llm = None if args.no_llm else "auto"
    if args.model or args.base_url:
        from llm import LLMClient
        import os as _os
        llm = LLMClient(
            api_key=_os.environ.get("OPENAI_API_KEY", ""),
            base_url=args.base_url or _os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=args.model or _os.environ.get("LLM_MODEL", "gpt-4o-mini"))
        if not llm.api_key and "localhost" not in llm.base_url \
                and "127.0.0.1" not in llm.base_url:
            llm = None  # hosted provider, no key: stay rule-based
    agent = Agent(args.db, args.notes, user_id=args.user, llm=llm,
                  mem0=None if args.no_mem0 else "auto")
    try:
        if args.ask is not None:
            reply, _ = agent.respond(args.ask)
            print(reply if reply != "__EXIT__" else "Goodbye.")
            return 0
        print("Luna-agent: a tiny self-learning agent. Type 'help', or 'exit' to quit.")
        print(agent.mode)
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
