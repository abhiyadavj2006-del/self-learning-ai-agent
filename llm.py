"""Optional LLM client for the self-learning agent -- stdlib only.

The video's demo loop calls an LLM (GPT-4o mini) with the retrieved
memories in the system prompt, then writes the conversation back to
memory. This module is that LLM side, with no new dependency: it
talks to any OpenAI-compatible ``/v1/chat/completions`` endpoint
with ``urllib``.

No key is needed to run the agent: when no LLM is configured,
``get_llm_client()`` returns ``None`` and the agent stays in its
rule-based mode. Configure with env vars (never hardcode a key):

    OPENAI_API_KEY   OpenAI key (or any provider key for a custom base URL)
    OPENAI_BASE_URL  default https://api.openai.com/v1
                     set to http://localhost:11434/v1 for a local
                     Ollama model -- then no key is needed at all
    LLM_MODEL        default gpt-4o-mini (the model in the video's demo)
    LLM_ENABLED=0    force rule-based mode even if a key is set

The prompts mirror what the video shows of Mem0's own prompts:
a MEMORY_ANSWER_PROMPT (answer from retrieved memories), a
FACT_RETRIEVAL_PROMPT (extract durable facts from a conversation)
and the idea of a custom update prompt (see README).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "gpt-4o-mini"  # the model in the video's memory demo
DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Mirrors Mem0's MEMORY_ANSWER_PROMPT (as described in the video and
# the ai-cookbook README): answer from the memories, say so honestly
# when nothing relevant was retrieved.
MEMORY_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with long-term memory. "
    "Use the memories listed below to personalise your answer. "
    "If a memory contradicts something the user just said, the most "
    "recent statement wins. If no memory is relevant, answer normally "
    "and say nothing about memory. Keep answers short and direct."
)

# Mirrors Mem0's FACT_RETRIEVAL_PROMPT (as described in the video):
# pull out personal preferences, details, plans, activities, health
# information and professional details as a JSON list.
FACT_EXTRACTION_PROMPT = (
    "Extract the durable facts about the user from this conversation. "
    "Look for personal preferences, details, plans, activities, health "
    "information and professional details. Ignore greetings, questions "
    "and anything already phrased as a question. "
    "Return ONLY a JSON list like: "
    '[{"key": "name", "value": "Dave", "negated": false}, '
    '{"key": "likes", "value": "pizza", "negated": true}]. '
    'Use short keys ("name", "likes", "plan", "identity"). '
    'Set "negated" to true when the user says they no longer like, '
    "do, or want something. If there are no durable facts, return []."
)


class LLMClient:
    """Minimal chat-completions client (OpenAI-compatible)."""

    def __init__(self, api_key: str | None = None,
                 base_url: str | None = None,
                 model: str | None = None, timeout: int = 30):
        self.api_key = api_key or ""
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.last_error: str | None = None

    @property
    def provider(self) -> str:
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return "local"
        if "openai" in self.base_url:
            return "openai"
        return "custom"

    def generate(self, messages: list[dict], system: str | None = None,
                 temperature: float = 0.3) -> str | None:
        """Return the assistant's reply text, or None on any failure
        (network, auth, bad response) -- the caller falls back to
        rule-based mode, it never crashes the agent."""
        payload_messages: list[dict] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages)
        body = json.dumps({
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            content = (data.get("choices") or [{}])[0].get("message", {}) \
                .get("content")
            return content.strip() if content else None
        except Exception as e:  # network/auth/timeout/parse: fall back
            self.last_error = f"{type(e).__name__}: {e}"
            return None


def _is_local_url(url: str) -> bool:
    return ("localhost" in url or "127.0.0.1" in url
            or url.startswith("http://192.168.")
            or url.startswith("http://10."))


def get_llm_client() -> LLMClient | None:
    """Build the client from env vars, or None when no LLM is
    configured (the agent then stays rule-based)."""
    if os.environ.get("LLM_ENABLED", "1") == "0":
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not _is_local_url(base_url):
        return None  # hosted provider with no key: stay rule-based
    return LLMClient(api_key=api_key, base_url=base_url,
                     model=os.environ.get("LLM_MODEL", DEFAULT_MODEL))


def extract_facts_with_llm(messages: list[dict],
                           llm: LLMClient) -> list[dict] | None:
    """Phase-1 extraction via LLM (what Mem0 does with its
    FACT_RETRIEVAL_PROMPT). Returns a list of
    {key, value, negated} dicts, or None if the LLM failed or found
    nothing -- the caller then falls back to the rule extractor."""
    reply = llm.generate(messages, system=FACT_EXTRACTION_PROMPT,
                         temperature=0.0)
    if not reply:
        return None
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(reply[start:end + 1])
    except (ValueError, TypeError):
        return None
    facts = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip().lower()
        value = str(item.get("value", "")).strip()
        if key and value:
            facts.append({"key": key, "value": value,
                          "negated": bool(item.get("negated", False))})
    return facts or None
