# self-learning-ai-agent

A minimal but **real, working** self-learning AI agent in Python — stdlib +
SQLite only. No LLM API key, no paid services, no Docker, no network calls
to paid APIs.

It remembers facts you tell it, recalls them after a restart, corrects
itself when you correct it, and has two tools (calculator, note search).
It ships as a command-line REPL **and** a small web UI (stdlib
`http.server` only).

> **Repo status:** this folder is a standalone GitHub repo in its own
> right — suggested repo name **`self-learning-ai-agent`**. It is one of
> 5 separate repos (not a monorepo). GitHub is not connected yet, so
> nothing here has been uploaded or pushed; see
> [Going live / publishing](#going-live--publishing) for the steps
> once GitHub is connected.

## What it does

The learning loop is small and real:

1. **Observe** — every turn is logged to the `interactions` table.
2. **Learn** — `remember that my favourite colour is blue`, `my name is
   Ada`, `I like hiking` store a `(key, value)` fact in SQLite.
3. **Recall** — `what is my favourite colour?` reads it back, even
   after the process exits and restarts, because the `.db` file is the
   memory.
4. **Correct** — telling it the same key again overwrites the old
   value, so `no, my favourite colour is green` is learned as a
   correction.

Tools (in `tools.py`):

| Tool | What it does |
|---|---|
| Calculator | Safe arithmetic via `ast` — no `eval()`/`exec()`. `calculate 12 * (3 + 4)` → `84`. Rejects names, attribute access, imports. |
| Note search | Full-text search over `notes/*.txt` / `*.md`. `search notes for hiking` returns file, line number and snippet. |

## Folder structure

```
self-learning-ai-agent/
├── README.md            # this file
├── LICENSE              # MIT
├── .gitignore           # excludes *.db / *.sqlite, __pycache__, .env
├── agent.py             # CLI REPL + --ask one-shot mode; reasoning/dispatch
├── memory.py            # SQLite memory: facts + interaction log
├── tools.py             # calculator (safe AST) + note search
├── server.py            # web UI backend (stdlib http.server only)
├── web/
│   └── index.html       # chat UI: chat, remembered-facts panel, examples
├── notes/               # sample notes the search tool indexes
│   ├── hiking.txt
│   ├── project-ideas.md
│   └── python-tips.txt
└── tests/
    └── test_agent.py    # unittest suite (stdlib only)
```

There is no `requirements.txt`: there are no dependencies to install.
Python 3.11+ is enough (developed/tested on Python 3.12.3).

## How to run

### Command line

```bash
python3 agent.py
```

```
Luna-agent: a tiny self-learning agent. Type 'help', or 'exit' to quit.
you> remember that my favourite colour is blue
agent> Got it -- I'll remember that your favourite colour is blue.
you> calculate 12 * (3 + 4)
agent> 12 * (3 + 4) = 84
you> search notes for hiking
agent> Found 5 match(es) for 'hiking': ...
you> exit
Goodbye -- everything you taught me is saved.
```

Restart it and ask again — it remembers:

```bash
python3 agent.py
# you> what is my favourite colour?
# agent> Your favourite colour is blue.
```

One-shot mode (useful for scripts):

```bash
python3 agent.py --ask "calculate 7 * 6"
python3 agent.py --ask "what is my favourite colour?"
python3 agent.py --db /tmp/test.db --ask "remember that my dog's name is Rex"
```

Useful commands inside the REPL: `help`, `list facts`,
`what do you know about me?`, `forget my favourite colour`, `stats`,
`list notes`, `exit` / `quit`.

### Web UI (local first)

```bash
python3 server.py
# Self-learning agent web UI: http://127.0.0.1:8000
```

Open http://127.0.0.1:8000 — chat with the agent, see the
**Remembered facts** panel update live, and use the calculator via chat
(e.g. `calculate 12 * (3 + 4)`). The web UI and the CLI share the same
SQLite file (`agent_memory.db` by default), so a fact taught in the
browser is recalled by the CLI and vice versa.

Options:

```bash
PORT=8080 python3 server.py            # env var for the port
python3 server.py --port 8080 --host 0.0.0.0 --db agent_memory.db
```

Env vars: `PORT` (default `8000`), `HOST` (default `127.0.0.1`).
No API keys, no secrets, no other env vars are needed.

API (JSON, same behaviour as the CLI):

| Endpoint | Method | Body / response |
|---|---|---|
| `/` | GET | `web/index.html` |
| `/api/chat` | POST | `{"message": "..."}` → `{"reply": "...", "tool": "calculator" \| "note_search" \| null}` |
| `/api/facts` | GET | `{"facts": [["favourite colour", "blue"]]}` |
| `/api/stats` | GET | `{"facts": 1, "interactions": 4, "tools": {"calculator": 1}}` |

## What was tested

Tested 2026-11-21 on Python 3.12.3, by actually running the code
(not by inspection). All tests passed.

**1. Unit tests — 6/6 passed**

```bash
python3 -m unittest discover -s tests -v
# test_basic, test_unsafe_rejected, test_learn_restart_recall,
# test_tools_via_agent, test_finds_hiking, test_no_match ... ok
# Ran 6 tests in 0.006s — OK
```

**2. CLI end-to-end (fresh DB, two separate runs)**

Run 1 taught `favourite colour = blue` and `name = Ada`, ran
`calculate 12 * (3 + 4)` → `84`, `what is 45 + 23?` → `68`, and
`search notes for hiking` → 5 matches. Run 2 (new process, same DB)
recalled both facts, listed them with `what do you know about me?`,
learned the correction `green`, and `forget my name` made the next
recall honestly answer `I don't know your name yet`. Direct SQLite
check confirmed the stored row: `[('favourite colour', 'green')]`
with 13 logged interactions.

**3. Web UI end-to-end (live server on 127.0.0.1:18099, throwaway DB)**

- `GET /` → HTTP 200, the chat page.
- `POST /api/chat` `remember that my favourite colour is blue` →
  `Got it -- I'll remember that your favourite colour is blue.`
- `POST /api/chat` `calculate 12 * (3 + 4)` → `12 * (3 + 4) = 84`,
  `"tool": "calculator"`.
- `POST /api/chat` recall → `Your favourite colour is blue.`
- `POST /api/chat` `search notes for hiking` → 5 matches,
  `"tool": "note_search"`.
- `GET /api/facts` → `{"facts": [["favourite colour", "blue"]]}`;
  `GET /api/stats` → `{"facts": 1, "interactions": 4, ...}`.
- Bad input → HTTP 400 `{"error": "message is required"}`;
  unknown endpoint → HTTP 404.
- The CLI then recalled the fact taught via the web UI from the
  shared DB — memory persists across interfaces, not just restarts.

Two bugs found by the end-to-end run were fixed and re-tested:
the confirmation used to say "your *my* favourite colour", and
`what do you know about me?` (with `?`) fell through to the
fallback reply. Both now work.

## Limitations — what this is NOT

This is a starter, not a complete agent. Documented honestly:

- **Rule-based, not an LLM.** It only understands the patterns in
  `agent.py` (`remember that my X is Y`, `what is my X?`,
  `calculate ...`, `search notes for ...`). Anything else gets an
  honest "I don't know how to answer that yet" plus `help`.
  It cannot reason, summarise, or answer open questions.
- **Flat key–value memory.** One value per key, no relationships,
  no time-decay, no confidence scores, no semantic search — a fact
  is only found by its exact key (or a substring fallback).
- **Single user, single machine.** No accounts, no auth. The web UI
  has no login; anyone who can reach it can read and change its
  memory. Do not expose it publicly without adding auth.
- **SQLite file.** Fine locally; on hosted platforms the DB is lost
  on redeploy unless you attach a persistent volume (see below).
- **Notes are static files.** Search only covers `notes/` at
  startup path; adding notes means adding files, and there is no
  note-writing tool yet.

## Next steps to make it fuller

1. **Plug in an LLM** (no key is needed today; this is the main
   upgrade): add an `llm.py` with a tiny interface —
   `generate(prompt, context) -> str` — and call it from
   `Agent.respond()` as a fallback when no rule matches, passing
   the recalled facts as context. Implement it first against a
   local model (e.g. Ollama, `http://localhost:11434`) so nothing
   is paid for; then add a provider client (OpenAI/Anthropic) that
   reads its key from an env var, e.g. `OPENAI_API_KEY`, and keep
   the rule-based path as the offline fallback when the key is absent.
2. **Plug in Mem0 (or similar long-term memory):** keep SQLite as
   the local store, and mirror each `set_fact` into Mem0
   (`mem0.add(...)`) and each recall into `mem0.search(...)`.
   That adds semantic recall ("what's my favourite thing?" finding
   "favourite colour") without replacing the local DB. Needs a
   Mem0 API key / self-hosted Mem0 — do not commit keys; use env vars.
3. **More tools:** web search (read-only), a `write note` tool
   (append to `notes/`), reminders (`remind me ...` + a `due`
   table), and a file-read tool scoped to a safe directory.
4. **Memory v2:** confidence + timestamps per fact, `list facts`
   showing when each was learned, and expiry/confirmation for
   stale facts.
5. **Web UI v2:** delete-a-fact buttons (the API already exposes
   the facts), per-user sessions, and auth before any public deploy.

## Going live / publishing

**Nothing has been deployed or uploaded.** GitHub is not connected
yet, and no remote, credentials, or push were created. What is
still needed:

### Publish to GitHub (once GitHub is connected)

```bash
cd self-learning-ai-agent
git init
git add .
git commit -m "Self-learning AI agent: stdlib Python, SQLite memory, web UI"
# create the repo on GitHub first (name: self-learning-ai-agent),
# then, with the real URL:
git remote add origin https://github.com/<username>/self-learning-ai-agent.git
git push -u origin main
```

The `.gitignore` already excludes the SQLite DB (`*.db`, `*.sqlite`),
so personal facts never get committed. The repo needs no secrets.

### Deploy the web UI live

Any host that runs Python 3.11+ works — no build step, no
dependencies to install. Examples: **Render** (Web Service,
`python3 server.py`, free tier), **Railway** or **Fly.io**
(same start command), a **VPS** (`python3 server.py` behind
systemd/nginx), or **PythonAnywhere**. Commands are the same
everywhere:

```bash
python3 server.py            # listens on $PORT (default 8000)
```

Env vars: `PORT` (set by Render/Railway/Fly automatically —
the server already reads it), `HOST=0.0.0.0` (required on hosted
platforms so the platform can reach it; locally it defaults to
`127.0.0.1`). No API keys or other env vars.

**One caveat before going live:** on Render/Railway/Fly the
filesystem is ephemeral — `agent_memory.db` is wiped on each
redeploy/restart. For a live deployment, either attach a
persistent disk/volume and point `--db` at it
(`python3 server.py --db /data/agent_memory.db`), or move the
facts table to a hosted Postgres (the `Memory` class is the only
file that would change). Also add auth in front of the UI before
exposing it publicly — it currently has none.

## License

MIT — see [LICENSE](LICENSE).
