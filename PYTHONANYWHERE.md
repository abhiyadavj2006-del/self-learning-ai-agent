# Deploy on PythonAnywhere (free Beginner tier)

PythonAnywhere runs web apps through **WSGI**, not via
`python3 server.py`. `pythonanywhere_wsgi.py` (in this repo) adapts
the existing stdlib server to WSGI without changing the app.

## 1. Create the account

`pythonanywhere.com` -> "Create a Beginner account" (free, **no
card**). Pick the username carefully: your site will be
`https://USERNAME.pythonanywhere.com/`.

> **One free account = one web app.** A Beginner account hosts a
> single web app (confirmed on PythonAnywhere's own pricing page;
> even the $10/month Developer plan allows only one — only Custom
> plans allow more). To host both this agent and the search engine
> for free you need **two free accounts** (two subdomains), or one
> combined app (see "Two apps, one account" below), or a paid
> Custom plan.

## 2. Get the code on the server

Open a **Bash console** from the dashboard and clone this repo:

```bash
git clone https://github.com/abhiyadavj2006-del/self-learning-ai-agent.git
cd self-learning-ai-agent
python3 --version   # use the same version for the web app below
```

No dependencies to install (stdlib only), no virtualenv needed.

## 3. Create the web app

Web tab -> **Add a new web app** -> choose the free domain
(`USERNAME.pythonanywhere.com`) -> **Manual configuration** ->
pick **Python 3.12** (or the newest offered; the code needs 3.10+).

Then in the Web tab set:

| Setting          | Value                                              |
| ---------------- | -------------------------------------------------- |
| Source code      | `/home/USERNAME/self-learning-ai-agent`            |
| Working directory| `/home/USERNAME/self-learning-ai-agent`            |
| Virtualenv       | (leave blank — stdlib only, nothing to install)    |

## 4. Point the WSGI file at this repo

Click the **WSGI configuration file** link in the Web tab (a file
like `/var/www/USERNAME_pythonanywhere_com_wsgi.py`) and replace
its contents with:

```python
import sys
sys.path.insert(0, '/home/USERNAME/self-learning-ai-agent')
from pythonanywhere_wsgi import application
```

Replace `USERNAME` with your PythonAnywhere username. Then hit
**Reload** and visit `https://USERNAME.pythonanywhere.com/`.

## 5. Static files

None required — the app serves `web/index.html` itself. (If you
later add CSS/JS under `web/`, you may map `/static/` to that
folder in the Web tab's Static Files section, but it is optional.)

## What lives where (data)

- **Memory is a file, and it persists.** The agent's SQLite memory
  (`agent_memory.db`) is created in the repo folder on first use.
  PythonAnywhere's disk is persistent (512 MB on the free tier) —
  learned facts survive reloads and redeploys, unlike Render's
  ephemeral disk. The file is gitignored, so it never gets
  committed.
- **No auth.** Anyone with the URL can chat and teach the agent.
  Do not share the URL publicly unless you add a login in front
  of it.
- **LLM / Mem0 are off here.** Free accounts only allow outbound
  HTTP(S) to a small whitelist of sites, and `api.openai.com` is
  not on it, so the optional LLM and real-Mem0 backends can never
  answer from a free account. `pythonanywhere_wsgi.py` defaults
  `LLM_ENABLED=0` / `MEM0_ENABLED=0`, so the app runs in its
  rule-based mode (exactly like the no-key local runs). Remove
  those two lines only if you move to a paid plan and set a key.
- **CPU budget:** 100 CPU-seconds/day on the free tier. Each
  request opens the small SQLite DB; fine for demo traffic.

## Two apps, one account (options)

- **Two free accounts** (simplest, no code changes): one account
  for this agent, a second for the search engine. Two subdomains.
- **One combined app:** a single WSGI app that routes by path
  prefix (e.g. `/` -> agent, `/search` -> search engine). Needs a
  small dispatcher file — not written yet; ask for it if wanted.
- **Paid Custom plan** ($10+/month): up to 20 web apps on one
  account.

## Local test (no account needed)

```bash
python3 pythonanywhere_wsgi.py   # http://127.0.0.1:8000
```
