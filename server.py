#!/usr/bin/env python3
"""Web UI for the self-learning agent -- Python stdlib http.server only.

No framework, no API key, no network calls. Serves web/index.html and a
tiny JSON API backed by the same Agent / SQLite memory as agent.py, so
facts taught in the web UI are recalled by the CLI (and vice versa --
they share the same DB file).

Endpoints:
  GET  /            -> web/index.html (chat UI)
  GET  /api/facts   -> {"facts": [[key, value], ...]}
  GET  /api/stats   -> {"facts": n, "interactions": n, "tools": {...},
                       "llm": ..., "mem0": ..., "mode": ...}
  GET  /api/search?q=... -> {"results": [{"memory", "score", ...}]}
  POST /api/chat    {"message": "..."} -> {"reply": "...", "tool": "calculator"|...|null}
  Any other static file under web/ is served too (path-traversal safe).

Run locally:
  python3 server.py                 # http://127.0.0.1:8000
  PORT=8080 python3 server.py       # env var wins over the default
  python3 server.py --port 8080 --host 0.0.0.0 --db agent_memory.db

Deploy: any host that runs Python 3.11+ (see README "Going live").
Only env var: PORT (default 8000). HOST defaults to 127.0.0.1; set
HOST=0.0.0.0 when deploying so the platform can reach it.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from agent import Agent  # noqa: E402
from memory import DEFAULT_USER_ID  # noqa: E402

WEB_DIR = BASE / "web"
LOCK = threading.Lock()  # serialize SQLite access across threads


def make_handler(db_path: str, notes_dir: str,
                 user_id: str = DEFAULT_USER_ID,
                 llm="auto", mem0="auto"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stdout clean
            pass

        # -- helpers --------------------------------------------------
        def _json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _agent(self) -> Agent:
            return Agent(db_path, notes_dir, user_id=user_id,
                         llm=llm, mem0=mem0)

        # -- routes ---------------------------------------------------
        def do_GET(self):
            parts = urlsplit(self.path)
            if parts.path in ("/", "/index.html"):
                self._serve_file(WEB_DIR / "index.html")
            elif parts.path == "/api/facts":
                with LOCK:
                    a = self._agent()
                    try:
                        self._json({"facts": a.memory.list_facts(
                            user_id=user_id)})
                    finally:
                        a.close()
            elif parts.path == "/api/stats":
                with LOCK:
                    a = self._agent()
                    try:
                        stats = a.memory.stats()
                        stats["llm"] = (a.llm.model if a.llm else None)
                        stats["mem0"] = (a.mem0.describe()
                                         if a.mem0 else "off")
                        stats["mode"] = a.mode
                        self._json(stats)
                    finally:
                        a.close()
            elif parts.path == "/api/search":
                query = parse_qs(parts.query).get("q", [""])[0].strip()
                if not query:
                    self._json({"error": "q is required"}, 400)
                    return
                with LOCK:
                    a = self._agent()
                    try:
                        self._json({"results": a.memory.search(
                            query, user_id=user_id, limit=5)})
                    finally:
                        a.close()
            elif parts.path.startswith("/api/"):
                self._json({"error": "unknown endpoint"}, 404)
            else:
                # Static file under web/, traversal-safe.
                rel = Path(parts.path.lstrip("/"))
                target = (WEB_DIR / rel).resolve()
                if WEB_DIR.resolve() not in target.parents or not target.is_file():
                    self._json({"error": "not found"}, 404)
                else:
                    self._serve_file(target)

        def do_POST(self):
            if self.path != "/api/chat":
                self._json({"error": "unknown endpoint"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                message = str(payload.get("message", "")).strip()
            except (ValueError, OSError):
                self._json({"error": "invalid JSON"}, 400)
                return
            if not message:
                self._json({"error": "message is required"}, 400)
                return
            with LOCK:
                a = self._agent()
                try:
                    reply, tool = a.respond(message)
                finally:
                    a.close()
            self._json({"reply": "Goodbye." if reply == "__EXIT__" else reply,
                        "tool": tool})

        def _serve_file(self, path: Path) -> None:
            try:
                data = path.read_bytes()
            except OSError:
                self._json({"error": "not found"}, 404)
                return
            ctype, _ = mimetypes.guess_type(str(path))
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Web UI for the self-learning agent")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--db", default=str(BASE / "agent_memory.db"))
    p.add_argument("--notes", default=str(BASE / "notes"))
    p.add_argument("--user", default=DEFAULT_USER_ID,
                   help="user_id for memory scoping (default: default)")
    p.add_argument("--no-llm", action="store_true",
                   help="force rule-based mode even if a key is set")
    p.add_argument("--no-mem0", action="store_true",
                   help="disable the real-Mem0 mirror even if configured")
    args = p.parse_args(argv)

    llm = None if args.no_llm else "auto"
    mem0 = None if args.no_mem0 else "auto"
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(args.db, args.notes, user_id=args.user,
                     llm=llm, mem0=mem0))
    print(f"Self-learning agent web UI: http://{args.host}:{args.port}", flush=True)
    print(f"Memory: {args.db}  |  Notes: {args.notes}  (Ctrl-C to stop)", flush=True)
    _probe = Agent(args.db, args.notes, user_id=args.user,
                   llm=llm, mem0=mem0)
    print(_probe.mode, flush=True)
    _probe.close()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
