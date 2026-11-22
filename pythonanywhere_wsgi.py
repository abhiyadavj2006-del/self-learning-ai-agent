#!/usr/bin/env python3
"""WSGI entry point for PythonAnywhere (free Beginner tier).

PythonAnywhere web apps do NOT run via ``python3 server.py``: the
Web tab runs your code through a WSGI file, and ``app.run()`` /
``serve_forever()`` would crash the app. This file adapts the existing
stdlib ``http.server`` handler (``server.py``) to WSGI **without
changing the app at all**: each WSGI request is replayed through the
real ``BaseHTTPRequestHandler`` over a socketpair, and the raw HTTP
response is translated back into a WSGI ``(status, headers, body)``
triple. Routing, JSON, static files and the SQLite memory all run
unchanged.

Deploy (see PYTHONANYWHERE.md for the full steps):
  1. Clone the repo into your PythonAnywhere home dir, e.g.
     ``/home/USERNAME/self-learning-ai-agent``.
  2. Web tab -> Add a new web app -> Manual configuration ->
     Python 3.12 (or newer). No virtualenv needed (stdlib only).
  3. In the Web tab, open the WSGI configuration file and replace its
     contents with::

         import sys
         sys.path.insert(0, '/home/USERNAME/self-learning-ai-agent')
         from pythonanywhere_wsgi import application

     (replace USERNAME with your PythonAnywhere username).
  4. Set Source code / Working directory to the repo folder, hit
     Reload, and visit ``https://USERNAME.pythonanywhere.com/``.

Local smoke test (stdlib only, no PythonAnywhere needed)::

    python3 pythonanywhere_wsgi.py        # http://127.0.0.1:8000

Environment (set them in the WSGI file above if you need to change
them; PythonAnywhere's Web tab has no env-var UI):
  PA_PROJECT_DIR   repo folder (default: this file's folder)
  PA_AGENT_DB      SQLite memory file (default: agent_memory.db here)
  PA_AGENT_NOTES   notes folder (default: notes/ here)
  PA_AGENT_USER    user_id for memory scoping (default: default)
  LLM_ENABLED=0    defaulted below (see note)
  MEM0_ENABLED=0   defaulted below (see note)

Free-tier note: PythonAnywhere's free accounts only allow outbound
HTTP(S) to a small whitelist of sites -- api.openai.com is NOT on
it -- so the optional LLM / real-Mem0 backends can never answer
there. This file therefore defaults LLM_ENABLED=0 and
MEM0_ENABLED=0, so the app runs in its rule-based mode (which is
exactly what the no-key local runs do). Remove those two lines if
you ever deploy on a paid plan with a key set.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import quote_from_bytes

PROJECT_DIR = Path(os.environ.get("PA_PROJECT_DIR",
                                  Path(__file__).resolve().parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# See the free-tier note in the docstring above.
os.environ.setdefault("LLM_ENABLED", "0")
os.environ.setdefault("MEM0_ENABLED", "0")

from server import make_handler  # noqa: E402
from memory import DEFAULT_USER_ID  # noqa: E402

DB_PATH = os.environ.get("PA_AGENT_DB", str(PROJECT_DIR / "agent_memory.db"))
NOTES_DIR = os.environ.get("PA_AGENT_NOTES", str(PROJECT_DIR / "notes"))
USER_ID = os.environ.get("PA_AGENT_USER", DEFAULT_USER_ID)

_HANDLER_CLS = make_handler(DB_PATH, NOTES_DIR, user_id=USER_ID,
                            llm="auto", mem0="auto")


# Hop-by-hop headers must never reach a WSGI start_response().
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailer",
               "transfer-encoding", "upgrade"}


def _wsgi_from_handler(handler_cls):
    """Adapt a BaseHTTPRequestHandler subclass to a WSGI app.

    The handler code runs untouched: we build the exact HTTP request
    bytes a client would have sent, feed them to the handler through
    a socketpair, then parse the raw HTTP response it writes back.
    """
    def application(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        script = environ.get("SCRIPT_NAME", "") or ""
        path = environ.get("PATH_INFO", "") or "/"
        # PATH_INFO arrives URL-decoded (latin-1); re-encode it so the
        # handler sees the same percent-encoded path a client sent.
        target = quote_from_bytes(
            (script + path).encode("iso-8859-1", "replace"), safe="/%")
        query = environ.get("QUERY_STRING", "")
        if query:
            target += "?" + query
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except (TypeError, ValueError):
            length = 0
        body = environ["wsgi.input"].read(length) if length > 0 else b""
        req_lines = [f"{method} {target or '/'} HTTP/1.1",
                     "Host: localhost", "Connection: close"]
        if environ.get("CONTENT_TYPE"):
            req_lines.append(f"Content-Type: {environ['CONTENT_TYPE']}")
        if length:
            req_lines.append(f"Content-Length: {length}")
        for key, value in sorted(environ.items()):
            if key.startswith("HTTP_"):
                req_lines.append(
                    f"{key[5:].replace('_', '-').title()}: {value}")
        raw = ("\r\n".join(req_lines) + "\r\n\r\n").encode("iso-8859-1") + body
        try:
            client, server_sock = socket.socketpair()
            try:
                client.sendall(raw)
                client.shutdown(socket.SHUT_WR)
                handler = handler_cls.__new__(handler_cls)
                handler.connection = server_sock
                handler.rfile = server_sock.makefile("rb")
                handler.wfile = server_sock.makefile("wb")
                handler.client_address = ("127.0.0.1", 0)
                handler.server = None
                try:
                    handler.handle_one_request()
                finally:
                    try:
                        handler.wfile.flush()
                    except Exception:
                        pass
                    handler.wfile.close()
                    handler.rfile.close()
                    server_sock.close()
                chunks = []
                while True:
                    data = client.recv(65536)
                    if not data:
                        break
                    chunks.append(data)
            finally:
                client.close()
            head, _, resp_body = b"".join(chunks).partition(b"\r\n\r\n")
            lines = head.split(b"\r\n") or [b"HTTP/1.1 500 Internal Error"]
            try:
                status = lines[0].decode("iso-8859-1").split(" ", 1)[1]
            except IndexError:
                status = "500 Internal Server Error"
            headers = []
            for line in lines[1:]:
                if b":" in line:
                    key, value = line.split(b":", 1)
                    key = key.decode("iso-8859-1").strip()
                    # Hop-by-hop headers (e.g. "Connection: close", added
                    # by send_error) are illegal in a WSGI response --
                    # wsgiref and mod_wsgi both reject them -- so drop
                    # them; the WSGI server manages the connection.
                    if key.lower() in _HOP_BY_HOP:
                        continue
                    headers.append((key,
                                    value.decode("iso-8859-1").strip()))
        except Exception as exc:  # never leak a bare socket error to WSGI
            import json
            resp_body = json.dumps(
                {"error": f"wsgi bridge failed: {exc}"}).encode()
            status = "500 Internal Server Error"
            headers = [("Content-Type", "application/json"),
                       ("Content-Length", str(len(resp_body)))]
        start_response(status, headers)
        return [resp_body]

    return application


application = _wsgi_from_handler(_HANDLER_CLS)


if __name__ == "__main__":
    # Local smoke test only -- PythonAnywhere imports `application`.
    from wsgiref.simple_server import make_server
    port = int(os.environ.get("PORT", "8000"))
    print(f"WSGI app (local test only): http://127.0.0.1:{port}")
    make_server("127.0.0.1", port, application).serve_forever()
