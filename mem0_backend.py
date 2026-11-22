"""Optional real-Mem0 backend -- the video's library, used as-is.

The video shows two ways to run Mem0:

* Cloud: ``MemoryClient(api_key=...)`` from app.mem0.ai
  (``MEM0_API_KEY`` env var).
* Open source: ``Memory()`` / ``Memory.from_config(config)`` with a
  vector store -- Qdrant via ``docker compose up -d`` in the video
  (``QDRANT_HOST`` / ``QDRANT_PORT`` env vars; see docker-compose.yml).

This module wraps either one behind the same ``add`` / ``search`` /
``get_all`` calls the local SQLite memory uses, so the agent can
mirror every fact into real Mem0 when it is available -- and keep
working on local SQLite alone when it is not. Nothing here is
required: ``mem0ai`` is an optional dependency (see
requirements-optional.txt), no key is hardcoded, and every failure
(import missing, bad key, network down) just disables the backend
instead of breaking the agent.

Env vars:
    MEM0_API_KEY   Mem0 Platform key (cloud). If set, cloud wins.
    MEM0_ENABLED=0 force-disable even if a key/package is present.
    QDRANT_HOST / QDRANT_PORT  for the open-source + Qdrant setup.
    OPENAI_API_KEY also needed for the open-source Memory (its LLM
                   and embedder default to OpenAI).
"""
from __future__ import annotations

import os


class Mem0Backend:
    """Thin wrapper around Mem0 Cloud (MemoryClient) or open-source
    Mem0 (Memory / Memory.from_config). ``available`` is False when
    Mem0 is not installed or not configured."""

    def __init__(self):
        self.available = False
        self.mode = "not configured"
        self._client = None
        self.last_error: str | None = None
        if os.environ.get("MEM0_ENABLED", "1") == "0":
            return
        try:
            self._init_client()
        except Exception as e:  # missing package, bad key, bad config
            self.last_error = f"{type(e).__name__}: {e}"
            self._client = None
            self.available = False
            self.mode = "not configured"

    def _init_client(self) -> None:
        api_key = os.environ.get("MEM0_API_KEY", "")
        if api_key:
            from mem0 import MemoryClient
            self._client = MemoryClient(api_key=api_key)
            self.available = True
            self.mode = "Mem0 Cloud"
            return
        # Open-source Mem0 -- needs the package and an LLM key.
        from mem0 import Memory
        qdrant_host = os.environ.get("QDRANT_HOST", "")
        if qdrant_host:
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": qdrant_host,
                        "port": int(os.environ.get("QDRANT_PORT", "6333")),
                    },
                },
                "llm": {"provider": "openai",
                        "config": {"api_key": os.environ["OPENAI_API_KEY"]}},
                "embedder": {"provider": "openai",
                             "config": {
                                 "api_key": os.environ["OPENAI_API_KEY"],
                                 "model": "text-embedding-3-small"}},
            }
            self._client = Memory.from_config(config)
            self.mode = "Mem0 OSS + Qdrant"
        else:
            self._client = Memory()
            self.mode = "Mem0 OSS (in-memory)"
        self.available = True

    # -- passthroughs (Mem0-shaped in, best-effort out) ----------------
    def add(self, messages, user_id: str = "default") -> bool:
        if not self.available:
            return False
        try:
            self._client.add(messages, user_id=user_id)
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    def search(self, query: str, user_id: str = "default",
               limit: int = 3) -> list:
        if not self.available:
            return []
        try:
            res = self._client.search(query, user_id=user_id, limit=limit)
            return res if isinstance(res, list) else []
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return []

    def get_all(self, user_id: str = "default") -> list:
        if not self.available:
            return []
        try:
            res = self._client.get_all(user_id=user_id)
            return res if isinstance(res, list) else []
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return []

    def describe(self) -> str:
        return self.mode if self.available else "not configured"


def get_mem0_backend() -> Mem0Backend:
    return Mem0Backend()
