"""Automated tests for the self-learning agent. Stdlib only (unittest).

Run:  python3 -m unittest discover -s tests -v
Covers: fact learn/recall/forget + correction, calculator (incl. rejection
of unsafe input), note search, and SQLite persistence across two separate
Agent instances (simulating exit + restart).
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent  # noqa: E402
from tools import calculate, search_notes  # noqa: E402


class TestCalculator(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(calculate("12 * (3 + 4)"), 84)
        self.assertEqual(calculate("45 + 23"), 68)
        self.assertEqual(calculate("10 / 4"), 2.5)

    def test_unsafe_rejected(self):
        for bad in ["__import__('os').system('id')", "1; import os", "[1,2][0]"]:
            with self.assertRaises(ValueError, msg=bad):
                calculate(bad)


class TestNoteSearch(unittest.TestCase):
    def test_finds_hiking(self):
        hits = search_notes("hiking", Path(__file__).parent.parent / "notes")
        self.assertTrue(any("hiking" in h["text"].lower() for h in hits))
        self.assertTrue(all(h["file"] for h in hits))

    def test_no_match(self):
        self.assertEqual(
            search_notes("zzzzqqq", Path(__file__).parent.parent / "notes"), [])


class TestMemoryPersistence(unittest.TestCase):
    def test_learn_restart_recall(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "mem.db")
            notes = str(Path(__file__).parent.parent / "notes")
            a1 = Agent(db, notes)
            reply, _ = a1.respond("remember that my favourite colour is blue")
            self.assertIn("blue", reply)
            a1.close()  # <-- simulate exit

            a2 = Agent(db, notes)  # <-- simulate restart, same DB file
            reply, _ = a2.respond("what is my favourite colour?")
            self.assertIn("blue", reply)
            # correction overwrites
            reply, _ = a2.respond("no, my favourite colour is green")
            self.assertIn("green", reply)
            reply, _ = a2.respond("what is my favourite colour?")
            self.assertIn("green", reply)
            # forget
            reply, _ = a2.respond("forget my favourite colour")
            self.assertIn("Forgot", reply)
            reply, _ = a2.respond("what is my favourite colour?")
            self.assertIn("don't know", reply)
            a2.close()

    def test_tools_via_agent(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "mem.db")
            notes = str(Path(__file__).parent.parent / "notes")
            a = Agent(db, notes)
            reply, tool = a.respond("calculate 12 * (3 + 4)")
            self.assertEqual(tool, "calculator")
            self.assertIn("84", reply)
            reply, tool = a.respond("search notes for hiking")
            self.assertEqual(tool, "note_search")
            self.assertIn("hiking", reply.lower())
            stats = a.memory.stats()
            self.assertEqual(stats["tools"].get("calculator"), 1)
            self.assertEqual(stats["tools"].get("note_search"), 1)
            a.close()


if __name__ == "__main__":
    unittest.main()
