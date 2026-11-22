# self-learning-ai-agent

A **working** self-learning AI agent in Python, built to follow
Dave Ebbelaar's tutorial on long-term memory for agents. It runs
with **zero dependencies** (stdlib + SQLite) and no API key — and it
can grow into the video's full setup (LLM + real Mem0 + Qdrant) by
setting a couple of env vars.

It remembers facts you tell it, recalls them after a restart,
corrects itself when you correct it (including contradictions like
"I don't like pizza anymore"), and has two tools (calculator, note
search). It ships as a command-line REPL **and** a small web UI
(stdlib `http.server` only).

> **Repo status:** this folder is a standalone GitHub repo in its own
> right — suggested repo name **`self-learning-ai-agent`**. It is one of
> 5 separate repos (not a monorepo). It was already uploaded to GitHub
> (`abhiyadavj2006-del/self-learning-ai-agent`) with GitHub Pages live;
> this update is local only and has **not** been pushed — see
> [Going live / publishing](#going-live--publishing).

## Source — what this is based on

**Video:** "How to Build Self-Learning AI Agents (Python Tutorial)"
by **Dave Ebbelaar** (@daveebbelaar), published 8 May 2025 —
https://www.youtube.com/watch?v=ynhl8KjjS3Y

**Companion repo (the video's code):**
https://github.com/daveebbelaar/ai-cookbook/tree/main/knowledge/mem0

**Library:** Mem0 — https://mem0.ai — docs at https://docs.mem0.ai

How the video was used for this update (honest note): the video was
not just skimmed from its description. The **full English
auto-caption transcript** (~4,700 words) was read end to end, plus
the video description/timestamps and the companion repo's README
(which documents the same code the video walks through). No full
video watch was possible in this environment; where the transcript
was unclear, the companion repo and Mem0 docs were used to confirm
details. The transcript itself is not shipped in this repo.

### What the video teaches

1. **Why memory matters:** plain LLMs forget everything between
   chats. ChatGPT's memory feature (Settings → Personalization) is
   the reference: it notices a core memory, calls a tool, and stores
   it — the agent should do the same.
2. **Mem0's two-phase pipeline** (the core idea): given a
   conversation, first **summarise + extract** the salient facts,
   then an **update phase** decides per fact whether to **ADD**,
   **UPDATE**, **DELETE** or do nothing (**NONE**) — e.g. a
   contradiction deletes the old memory. This is smarter than
   dumping whole chats into a vector DB and semantic-searching them.
3. **The demo loop** (the OSS `memory_demo.py` in the companion
   repo): every chat turn, `memory.search(query, user_id, limit=3)`
   retrieves the relevant memories, they are printed and injected
   into the system prompt, the LLM (GPT-4o mini) answers, and then
   `memory.add(messages, user_id)` writes the turn back to memory.
4. **Cloud vs open source:** Mem0 Cloud (`MemoryClient`, API key,
   dashboard) vs open-source `Memory()` — same `add` / `search` /
   `get_all` interface. OSS without a vector store keeps memories
   only in memory (gone when the session ends).
5. **Persistence with Qdrant:** `docker compose up -d` for Qdrant,
   then `Memory.from_config(...)` with `vector_store: qdrant`,
   `llm: openai`, `embedder: openai (text-embedding-3-small)`.
   Config can also add a graph store (Mem0ᵍ / Neo4j) and
   `history_db_path`.
6. **Custom prompts:** the fact-extraction and update-memory prompts
   are plain text in the library and can be overridden in config —
   Dave recommends tailoring them to your use case.
7. **The pizza → pasta demo (and why it's not magic):** "I like
   pizza" is stored; "I don't like pizza anymore, I like pasta now"
   first leaves *both* pizza and pasta in memory (the agent answers
   "your favourite foods are pizza and pasta"), and only on the next
   pass does the update phase correctly delete pizza. An LLM decides
   these things, and it sometimes gets them wrong.
8. **Dave's challenge:** don't just use the library — understand its
   prompts and pipeline, and consider building the pieces yourself,
   because the abstraction layers hide the underlying vector DB and
   LLM methods you may later need.

## What it does

The learning loop, mirroring the video:

1. **Retrieve** — every turn, `memory.search(input, limit=3)` pulls
   the most relevant memories (relevance-scored; see
   [Limitations](#limitations--what-this-is-not)).
2. **Answer** — rule-based by default; with an LLM configured, the
   retrieved memories are injected into the system prompt and the LLM
   answers (the video's demo loop, `Agent.chat_with_memories()`).
3. **Learn** — `remember that my favourite colour is blue`, `my name
   is Ada`, `I like hiking` store a fact in SQLite. Every statement
   is also mined for facts after the turn (the write-back step), so
   `I don't like pizza anymore, I like pasta now` is learned even
   though no rule matches that sentence.
4. **Correct** — telling it the same key again overwrites the old
   value; a contradiction *deletes* the contradicted memory
   (pizza → pasta) and records the dislike.
5. **History** — every ADD / UPDATE / DELETE is logged, so `history
   likes` shows the whole story of a memory, like Mem0's `history()`.

Memory is **multi-level**, like Mem0: `user_id` (long-term),
`session_id` (short-term) and `agent_id` (agent long-term) scopes,
plus a `category` per fact (preference, identity, plan, activity,
health, professional, fact) — the categories Mem0's
fact-retrieval prompt asks for. Use `--user alice` to keep two
people's memories separate.

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
├── .env.example         # optional keys (LLM / Mem0 / Qdrant) -- no real keys
├── requirements-optional.txt  # mem0ai etc. (nothing required to run)
├── docker-compose.yml   # Qdrant, for the open-source Mem0 + Qdrant setup
├── agent.py             # CLI REPL + --ask mode; dispatch + demo loop
├── memory.py            # Mem0-style memory: add/search/get_all/history
├── llm.py               # optional OpenAI-compatible LLM client (stdlib)
├── mem0_backend.py      # optional real-Mem0 mirror (cloud or OSS)
├── tools.py             # calculator (safe AST) + note search
├── server.py            # web UI backend (stdlib http.server only)
├── web/
│   └── index.html       # chat UI: chat, facts panel, memory search
├── notes/               # sample notes the search tool indexes
│   ├── hiking.txt
│   ├── project-ideas.md
│   └── python-tips.txt
└── tests/
    ├── test_agent.py          # original suite (6 tests)
    └── test_memory_advanced.py  # Mem0-style API, pipeline, LLM loop (16 tests)
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
Memory: SQLite (Mem0-style) | LLM: off (rule-based) | Mem0: off | user: default
Memory: agent_memory.db  |  Notes: notes
you> remember that my favourite colour is blue
agent> Got it -- I'll remember that your favourite colour is blue.
you> calculate 12 * (3 + 4)
agent> 12 * (3 + 4) = 84
you> search notes for hiking
agent> Found 5 match(es) for 'hiking': ...
you> I like pizza
agent> Got it -- I'll remember that your likes is pizza.
you> I had pizza yesterday, but I don't like it anymore. I like pasta now.
agent> Noted -- I've updated what I remember:
  - Delete: likes pizza
  - Add: does not like pizza
  - Add: likes pasta
you> exit
Goodbye -- everything you taught me is saved.
```

Restart it and ask again — it remembers:

```bash
python3 agent.py
# you> what is my favourite colour?
# agent> Your favourite colour is blue.
# you> what's my favourite food?
# agent> I don't have an exact fact for 'favourite food', but the closest
#        memories are:
#          - likes pasta (score 1.0)
#          - does not like pizza (score 1.0)
```

One-shot mode (useful for scripts):

```bash
python3 agent.py --ask "calculate 7 * 6"
python3 agent.py --ask "what is my favourite colour?"
python3 agent.py --db /tmp/test.db --ask "remember that my dog's name is Rex"
python3 agent.py --user alice --ask "remember that my name is Alice"
```

Useful commands inside the REPL: `help`, `list facts` / `memories`,
`search memory <query>` (relevance-ranked, with scores),
`history <key>` (ADD/UPDATE/DELETE log), `forget my favourite colour`,
`stats`, `mode`, `list notes`, `exit` / `quit`.

### Web UI (local first)

```bash
python3 server.py
# Self-learning agent web UI: http://127.0.0.1:8000
```

Open http://127.0.0.1:8000 — chat with the agent, see the
**Remembered facts** panel update live, use the **Search memory**
box, and use the calculator via chat (e.g. `calculate 12 * (3 + 4)`).
The web UI and the CLI share the same SQLite file
(`agent_memory.db` by default), so a fact taught in the browser is
recalled by the CLI and vice versa.

Options:

```bash
PORT=8080 python3 server.py            # env var for the port
python3 server.py --port 8080 --host 0.0.0.0 --db agent_memory.db
python3 server.py --user alice --no-llm
```

Env vars: `PORT` (default `8000`), `HOST` (default `127.0.0.1`).

API (JSON, same behaviour as the CLI):

| Endpoint | Method | Body / response |
|---|---|---|
| `/` | GET | `web/index.html` |
| `/api/chat` | POST | `{"message": "..."}` → `{"reply": "...", "tool": "calculator" \| "note_search" \| null}` |
| `/api/facts` | GET | `{"facts": [["favourite colour", "blue"]]}` |
| `/api/search?q=...` | GET | `{"results": [{"memory": "likes pasta", "score": 1.0, ...}]}` |
| `/api/stats` | GET | `{"facts": 1, "interactions": 4, "tools": {"calculator": 1}, "llm": ..., "mem0": ..., "mode": ...}` |

## Going further: LLM, real Mem0, Qdrant

Everything below is **optional**. With nothing set, the agent is
fully working and rule-based. No key is ever hardcoded or committed;
copy `.env.example` to `.env` (gitignored) or export the vars.

### 1. LLM answers (the video's demo loop)

```bash
export OPENAI_API_KEY="sk-..."        # from platform.openai.com
python3 agent.py                       # mode line now shows the LLM
```

With a key set, open questions are answered by the LLM with the
retrieved memories in the system prompt, and the turn is written back
to memory via LLM fact extraction (Mem0's FACT_RETRIEVAL_PROMPT
idea). Without a key the agent stays rule-based and says so honestly.

Free/local alternative — Ollama, no key needed:

```bash
ollama pull llama3.1                   # one-time, on your machine
export OPENAI_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.1
python3 agent.py
```

Other knobs: `LLM_MODEL` (default `gpt-4o-mini`, the video's demo
model), `LLM_ENABLED=0` / `--no-llm` to force rule-based mode.
`llm.py` uses stdlib `urllib` against any OpenAI-compatible
`/v1/chat/completions` endpoint, so no `openai` package is needed.

### 2. Real Mem0 mirror (cloud or open source)

```bash
pip install mem0ai                     # see requirements-optional.txt
export MEM0_API_KEY="..."              # cloud, from app.mem0.ai
python3 agent.py                       # every fact is mirrored to Mem0
```

Without `MEM0_API_KEY` but with `mem0ai` + `OPENAI_API_KEY`
installed, the open-source `Memory()` is used instead (in-memory
unless Qdrant is configured). `MEM0_ENABLED=0` / `--no-mem0`
disables the mirror. If the package or key is missing, the backend
silently stays off — local SQLite is always the source of truth.

### 3. Qdrant (persistent vector store for open-source Mem0)

```bash
docker compose up -d                   # starts Qdrant on :6333
export QDRANT_HOST=localhost
python3 agent.py
```

This mirrors the video: Mem0 is then configured with
`vector_store: qdrant` (plus OpenAI LLM + `text-embedding-3-small`
embedder) instead of in-memory storage. Check the dashboard at
http://localhost:6333/dashboard

## What was tested

Tested 2026-11-22 on Python 3.12.3, by actually running the code
(not by inspection).

**1. Unit tests — 22/22 passed** (was 6/6 before this update; the
original 6 still pass unchanged)

```bash
python3 -m unittest discover -s tests -v
# Ran 22 tests in 0.030s -- OK
```

New coverage (`tests/test_memory_advanced.py`, 16 tests): the video's
movie example (3 facts extracted), ADD/UPDATE/DELETE/NONE events +
`history()`, the pizza → pasta contradiction, relevance search
(scores + `limit=3`), user/agent/session scoping, `get`/`update`/
`delete` by id, categories, the LLM demo loop with a fake LLM (no
network, no key), no-key fallbacks, and the `search memory` /
`history` / `mode` REPL commands.

**2. CLI end-to-end (fresh DB)** — the pizza → pasta demo from the
video works: `I like pizza`, then `I had pizza yesterday, but I
don't like it anymore. I like pasta now.` → memories become
`likes pasta` + `does not like pizza`; `search memory favourite
food` returns both with score 1.0; `history likes` shows
ADD → DELETE → ADD. The movie example extracts all three facts
(plan, likes sci-fi, dislikes thriller). An old-format DB (starter
schema, `key` as primary key) migrates in place and its facts are
still recalled.

**3. Web UI end-to-end (live server, throwaway DB)** — `GET /` →
200; `POST /api/chat` learn/calculate/recall as before;
`GET /api/search?q=favourite%20food` → scored results;
`GET /api/stats` → includes `llm`/`mem0`/`mode`; bad input → 400,
unknown endpoint → 404 (unchanged).

**4. No-key behaviour verified** — with no env vars set, `LLM: off
(rule-based)`, `Mem0: off`, and the agent answers open questions
with the honest "I don't know how to answer that yet" fallback
(plus the tip about `OPENAI_API_KEY`). No real key, token or secret
exists anywhere in this repo.

## Limitations — what this is NOT

Documented honestly:

- **Relevance search is keyword-based, not semantic.** Mem0 uses
  vector embeddings (`text-embedding-3-small`) so "favourite food"
  finds "likes pasta" by meaning. Here the same query works through
  a small synonym bridge + token overlap + recency. Paraphrases
  outside that bridge ("what do I enjoy eating?") will miss. Real
  semantic search needs an embedder — that is what the real-Mem0
  mirror (§2 above) is for.
- **Rule-based fact extraction is narrow.** It understands `my X is
  Y`, `I like/love X`, `I don't like X anymore`, `I'm planning to
  X`, `I am X`. The video's extraction is an LLM call, so it handles
  free-form sentences ("Deshraj is building a Gmail AI agent") and
  pronoun resolution ("I don't like it anymore" → pizza) far better.
  With an LLM configured, this agent uses the LLM extractor too;
  without one, unresolvable pronouns are dropped rather than
  guessed — and, exactly as the video shows, the first pass of a
  contradiction can be imperfect.
- **No graph memory.** Mem0ᵍ (Neo4j graph store) is not implemented
  or mirrored; relationships between memories are not modelled.
- **No LLM reasoning without a key.** The rule-based mode only
  understands the patterns in `agent.py`. It cannot reason,
  summarise, or answer open questions.
- **Flat facts, one value per key** (except `likes`/`dislikes`,
  which are sets). No time-decay, no confidence scores, no
  expiry/confirmation for stale facts.
- **Single machine, no auth.** The web UI has no login; anyone who
  can reach it can read and change its memory. Do not expose it
  publicly without adding auth.
- **SQLite file.** Fine locally; on hosted platforms the DB is lost
  on redeploy unless you attach a persistent volume (see below).

## What was intentionally left out (and why)

- **Real embeddings / vector search in the local store** — needs an
  embedding model + API key or a local model download; the
  keyword + synonym search covers the video's own demo queries,
  and the real-Mem0 path provides true semantic search when wanted.
- **Mem0ᵍ graph store / Neo4j** — needs a Neo4j instance and paid
  hosting; the video only mentions it as an option.
- **Mem0 Cloud as the primary store** — needs an account + key;
  local SQLite stays the source of truth so the project works
  offline and the mirror is opt-in.
- **The video's MCP-server follow-up** — Dave points to a separate
  MCP crash-course video; out of scope for this project.
- **Full transcript in the repo** — the auto-caption transcript was
  used as a research source, not shipped (it is YouTube's
  auto-generated captions of Dave's video).

## Next steps to make it fuller

1. **Embeddings locally:** swap the keyword scorer for a local
   embedding model (e.g. via Ollama or `sentence-transformers`) so
   `search()` becomes true semantic search without a paid key.
2. **More tools:** web search (read-only), a `write note` tool
   (append to `notes/`), reminders (`remind me ...` + a `due`
   table), and a file-read tool scoped to a safe directory.
3. **Memory v2:** confidence + per-fact timestamps in `list facts`,
   expiry/confirmation for stale facts, and a `forget everything`
   command (Mem0's `delete_all`).
4. **Web UI v2:** delete-a-fact buttons (the API already exposes
   the facts), per-user sessions, and auth before any public deploy.
5. **MCP server:** expose the memory as an MCP server so other
   apps can use it, following Dave's MCP crash course.

## Going live / publishing

The repo was already published to GitHub and GitHub Pages is live
for it. **This update has not been pushed** — push it from the
machine where the repo lives:

```bash
cd self-learning-ai-agent
git add .
git commit -m "Mem0-style memory: two-phase pipeline, LLM loop, history, scopes"
git push
```

The `.gitignore` already excludes the SQLite DB (`*.db`, `*.sqlite`)
and `.env`, so personal facts and keys never get committed. The repo
needs no secrets to run.

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
`127.0.0.1`). Optional: `OPENAI_API_KEY` / `MEM0_API_KEY` if you
want the LLM / Mem0 features live — set them in the host's env-var
settings, never in the repo.

**Two caveats before going live:** on Render/Railway/Fly the
filesystem is ephemeral — `agent_memory.db` is wiped on each
redeploy/restart. For a live deployment, either attach a
persistent disk/volume and point `--db` at it
(`python3 server.py --db /data/agent_memory.db`), or move the
facts table to a hosted Postgres (the `Memory` class is the only
file that would change). Also add auth in front of the UI before
exposing it publicly — it currently has none.

## License

MIT — see [LICENSE](LICENSE).
