#!/usr/bin/env python3
"""Web UI for the self-learning agent -- Python stdlib http.server only.

No framework, no API key, no network calls. Serves web/index.html and a
tiny JSON API backed by the same Agent / SQLite memory as agent.py, so
facts taught in the web UI are recalled by the CLI (and vice versa --
they share the same DB file).

Endpoints:
  GET  /            -> web/index.html (chat UI)
  GET  /api/facts   -> {"facts": [[key, value], ...]}
  GET  /api/stats   -> {"facts": n, "interactions": n, "tools": {...}}
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

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from agent import Agent  # noqa: E402

WEB_DIR = BASE / "web"
LOCK = threading.Lock()  # serialize SQLite access across threads


def make_handler(db_path: str, notes_dir: str):
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
            return Agent(db_path, notes_dir)

        # -- routes ---------------------------------------------------
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._serve_file(WEB_DIR / "index.html")
            elif self.path == "/api/facts":
                with LOCK:
                    a = self._agent()
                    try:
                        self._json({"facts": a.memory.list_facts()})
                    finally:
                        a.close()
            elif self.path == "/api/stats":
                with LOCK:
                    a = self._agent()
                    try:
                        self._json(a.memory.stats())
                    finally:
                        a.close()
            elif self.path.startswith("/api/"):
                self._json({"error": "unknown endpoint"}, 404)
            else:
                # Static file under web/, traversal-safe.
                rel = Path(self.path.lstrip("/"))
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
    args = p.parse_args(argv)

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.db, args.notes))
    print(f"Self-learning agent web UI: http://{args.host}:{args.port}", flush=True)
    print(f"Memory: {args.db}  |  Notes: {args.notes}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
