"""Tests for the advanced, video-faithful memory + LLM loop.

Covers the Mem0-style API (add/search/get_all/get/update/delete/
history), the two-phase pipeline (ADD / UPDATE / DELETE / NONE),
the video's pizza -> pasta contradiction demo, the movie example,
user/agent/session scoping, the LLM demo loop (with a fake LLM --
no network, no key), and the no-key fallbacks. Stdlib only.

Run:  python3 -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent  # noqa: E402
from llm import get_llm_client  # noqa: E402
from mem0_backend import get_mem0_backend  # noqa: E402
from memory import Memory, extract_facts_from_text  # noqa: E402


class FakeLLM:
    """Stand-in LLM: records the prompts it got, returns canned text."""

    def __init__(self, reply="Sure thing.", facts_reply="[]"):
        self.reply = reply
        self.facts_reply = facts_reply
        self.seen_systems = []
        self.model = "fake-model"
        self.provider = "fake"

    def generate(self, messages, system=None, temperature=0.3):
        self.seen_systems.append(system or "")
        if system and "Extract the durable facts" in system:
            return self.facts_reply
        return self.reply


def _agent(tmpdir):
    return Agent(os.path.join(tmpdir, "mem.db"),
                 str(Path(__file__).parent.parent / "notes"),
                 llm=None, mem0=None)


class TestMem0StyleAdd(unittest.TestCase):
    def test_movie_example_from_video(self):
        """The video's OSS example: planning a movie, loves sci-fi,
        not a fan of thrillers -> three facts."""
        with tempfile.TemporaryDirectory() as td:
            m = Memory(os.path.join(td, "m.db"))
            res = m.add("I'm planning to watch a movie tonight. "
                        "I love sci-fi movies, but I'm not a big fan "
                        "of thriller movies.", user_id="alice")
            events = {r["memory"]: r["event"] for r in res["results"]}
            self.assertEqual(events.get("planning to watch a movie tonight"),
                             "ADD")
            self.assertEqual(events.get("likes sci-fi movies"), "ADD")
            self.assertEqual(events.get("does not like thriller movies"),
                             "ADD")
            self.assertEqual(m.get_fact("plan", user_id="alice"),
                             "watch a movie tonight")
            self.assertEqual(m.get_fact("likes", user_id="alice"),
                             "sci-fi movies")
            m.close()

    def test_add_update_delete_none_events(self):
        with tempfile.TemporaryDirectory() as td:
            m = Memory(os.path.join(td, "m.db"))
            r1 = m.add("my name is Dave", user_id="u1")["results"]
            self.assertEqual(r1[0]["event"], "ADD")
            r2 = m.add("my name is Dave", user_id="u1")["results"]
            self.assertEqual(r2[0]["event"], "NONE")
            r3 = m.add("my name is David", user_id="u1")["results"]
            self.assertEqual(r3[0]["event"], "UPDATE")
            self.assertEqual(m.get_fact("name", user_id="u1"), "David")
            # history() tracks the whole story of that memory id
            hist = m.history(r1[0]["id"])
            self.assertEqual([h["event"] for h in hist],
                             ["ADD", "UPDATE"])
            m.close()

    def test_pizza_pasta_contradiction(self):
        """The video's demo: like pizza, then 'don't like it anymore,
        like pasta now' -> pizza gone, pasta kept."""
        with tempfile.TemporaryDirectory() as td:
            m = Memory(os.path.join(td, "m.db"))
            m.add("I like pizza", user_id="u1")
            self.assertEqual(m.get_fact("likes", user_id="u1"), "pizza")
            m.add("I had pizza yesterday, but I don't like it anymore. "
                  "I like pasta now.", user_id="u1")
            self.assertEqual(m.get_fact("likes", user_id="u1"), "pasta")
            self.assertEqual(m.get_fact("dislikes", user_id="u1"), "pizza")
            m.close()

    def test_rule_extractor(self):
        facts = extract_facts_from_text(
            "my name is Ada. I like hiking. I don't like rain anymore.")
        by_key = {}
        for f in facts:
            by_key.setdefault(f["key"], []).append(f)
        self.assertEqual(by_key["name"][0]["value"], "Ada")
        positives = [f for f in by_key["likes"] if not f["negated"]]
        self.assertEqual(positives[0]["value"], "hiking")
        negs = [f for f in facts if f["negated"]]
        self.assertEqual(negs[0]["value"], "rain")
        # questions produce no facts
        self.assertEqual(extract_facts_from_text("what is my name?"), [])


class TestSearchAndScopes(unittest.TestCase):
    def test_search_scores_and_limit(self):
        with tempfile.TemporaryDirectory() as td:
            m = Memory(os.path.join(td, "m.db"))
            m.set_fact("name", "Dave", user_id="u1")
            m.set_fact("likes", "pizza, pasta", user_id="u1")
            m.set_fact("plan", "watch a movie tonight", user_id="u1")
            hits = m.search("what's my favourite food?", user_id="u1",
                            limit=3)
            self.assertTrue(hits)
            self.assertLessEqual(len(hits), 3)
            self.assertEqual(hits[0]["memory"], "likes pizza, pasta")
            self.assertTrue(0 < hits[0]["score"] <= 1.0)
            # name query finds the name memory, not the food one
            hits = m.search("what is my name", user_id="u1")
            self.assertEqual(hits[0]["memory"], "name is Dave")
            m.close()

    def test_user_scoping_isolated(self):
        with tempfile.TemporaryDirectory() as td:
            m = Memory(os.path.join(td, "m.db"))
            m.set_fact("name", "Alice", user_id="alice")
            m.set_fact("name", "Bob", user_id="bob")
            self.assertEqual(m.get_fact("name", user_id="alice"), "Alice")
            self.assertEqual(m.get_fact("name", user_id="bob"), "Bob")
            self.assertEqual(
                [f["value"] for f in m.get_all(user_id="alice")],
                ["Alice"])
            self.assertEqual(
                m.search("name", user_id="bob")[0]["memory"], "name is Bob")
            m.close()

    def test_agent_and_session_scopes(self):
        with tempfile.TemporaryDirectory() as td:
            m = Memory(os.path.join(td, "m.db"))
            m.add("Return short responses", agent_id="gmail-agent")
            m.add("I am building a Gmail AI agent", user_id="u1",
                  session_id="session-1")
            agent_mems = m.get_all(agent_id="gmail-agent")
            self.assertEqual(len(agent_mems), 1)
            self.assertEqual(agent_mems[0]["scope"], "agent")
            sess_mems = m.get_all(session_id="session-1")
            self.assertEqual(len(sess_mems), 1)
            self.assertEqual(sess_mems[0]["scope"], "session")
            m.close()

    def test_get_update_delete_by_id(self):
        with tempfile.TemporaryDirectory() as td:
            m = Memory(os.path.join(td, "m.db"))
            fid = m.add("my name is Dave")["results"][0]["id"]
            self.assertEqual(m.get(fid)["memory"], "name is Dave")
            self.assertTrue(m.update(fid, "David"))
            self.assertEqual(m.get(fid)["value"], "David")
            self.assertTrue(m.delete(fid))
            self.assertIsNone(m.get(fid))
            self.assertFalse(m.delete(999999))
            m.close()

    def test_categories(self):
        with tempfile.TemporaryDirectory() as td:
            m = Memory(os.path.join(td, "m.db"))
            m.add("my name is Dave. I like pizza. "
                  "I'm planning to watch a movie tonight.")
            cats = {f["key"]: f["category"]
                    for f in m.get_all()}
            self.assertEqual(cats["name"], "identity")
            self.assertEqual(cats["likes"], "preference")
            self.assertEqual(cats["plan"], "plan")
            m.close()


class TestLLMLoop(unittest.TestCase):
    def test_chat_with_memories_uses_retrieved_memories(self):
        """The video's demo loop: search(limit=3) -> memories into
        the system prompt -> LLM reply -> memory.add() write-back."""
        with tempfile.TemporaryDirectory() as td:
            a = _agent(td)
            fake = FakeLLM(reply="Hi Dave, nice to see you again.")
            a.llm = fake
            a.respond("remember that my name is Dave")
            reply = a.chat_with_memories("what is my name?")
            self.assertIn("Dave", reply)
            # the LLM saw the retrieved memory in its system prompt
            self.assertTrue(any("name is Dave" in s
                                for s in fake.seen_systems))
            a.close()

    def test_llm_extracts_facts_for_writeback(self):
        with tempfile.TemporaryDirectory() as td:
            a = _agent(td)
            fake = FakeLLM(
                reply="Pasta it is.",
                facts_reply='[{"key": "likes", "value": "pasta", '
                            '"negated": false}]')
            a.llm = fake
            a.chat_with_memories("I like pasta")
            self.assertEqual(a.memory.get_fact("likes"), "pasta")
            a.close()

    def test_no_llm_stays_rule_based(self):
        with tempfile.TemporaryDirectory() as td:
            a = _agent(td)
            self.assertIsNone(a.llm)
            # "xyzzy plugh" matches no rule, tool or recall pattern.
            reply, _ = a.respond("xyzzy plugh")
            self.assertIn("don't know how to answer", reply)
            a.close()

    def test_get_llm_client_needs_a_key(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY", "OPENAI_BASE_URL",
                            "LLM_MODEL", "LLM_ENABLED")}
        old = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(env)
            self.assertIsNone(get_llm_client())
        finally:
            os.environ.clear()
            os.environ.update(old)


class TestMem0BackendDisabled(unittest.TestCase):
    def test_no_package_no_key_is_disabled(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("MEM0_API_KEY", "MEM0_ENABLED",
                            "OPENAI_API_KEY", "QDRANT_HOST")}
        old = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(env)
            b = get_mem0_backend()
            self.assertFalse(b.available)
            self.assertEqual(b.search("hello"), [])
            self.assertFalse(b.add("hello"))
        finally:
            os.environ.clear()
            os.environ.update(old)


class TestAgentCommands(unittest.TestCase):
    def test_search_memory_history_mode_commands(self):
        with tempfile.TemporaryDirectory() as td:
            a = _agent(td)
            a.respond("remember that my favourite colour is blue")
            reply, _ = a.respond("search memory colour")
            self.assertIn("blue", reply)
            reply, _ = a.respond("history favourite colour")
            self.assertIn("ADD", reply)
            reply, _ = a.respond("mode")
            self.assertIn("LLM: off", reply)
            self.assertIn("Mem0: off", reply)
            reply, _ = a.respond("memories")
            self.assertIn("favourite colour is blue", reply)
            a.close()

    def test_compound_statement_learns_and_acknowledges(self):
        with tempfile.TemporaryDirectory() as td:
            a = _agent(td)
            a.respond("I like pizza")
            reply, _ = a.respond(
                "I had pizza yesterday, but I don't like it anymore. "
                "I like pasta now.")
            self.assertIn("pasta", reply)
            self.assertEqual(a.memory.get_fact("likes"), "pasta")
            a.close()


if __name__ == "__main__":
    unittest.main()
