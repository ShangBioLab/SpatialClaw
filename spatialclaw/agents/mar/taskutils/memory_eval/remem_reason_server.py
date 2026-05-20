"""Standalone ReMemR1-style reasoning service for external agent systems.

This service adapts the ReMemR1 memory revisit loop to a generic
"query + memory chunks" API so systems like SpatialClaw can call it as a
memory-reasoning operator.

Expected upstream model service:
- OpenAI-compatible `/v1/chat/completions`
- model already served separately (for example by vLLM / SGLang)

This service does not own model weights itself. It orchestrates the
ReMemR1-style update/recall/final-summary loop over candidate memories.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web


UPSTREAM_BASE_URL = os.getenv("REMEM_UPSTREAM_BASE_URL", os.getenv("URL", "http://127.0.0.1:8000/v1")).rstrip("/")
UPSTREAM_API_KEY = os.getenv("REMEM_UPSTREAM_API_KEY", os.getenv("API_KEY", "123-abc"))
DEFAULT_MODEL = os.getenv("REMEM_UPSTREAM_MODEL", "")
HOST = os.getenv("REMEM_REASON_HOST", "127.0.0.1")
PORT = int(os.getenv("REMEM_REASON_PORT", "8765"))
SERVICE_API_TOKEN = os.getenv("REMEM_REASON_API_TOKEN", "").strip()

DEFAULT_MAX_CHUNKS = int(os.getenv("REMEM_REASON_MAX_CHUNKS", "8"))
DEFAULT_MAX_MEMORY_TOKENS = int(os.getenv("REMEM_REASON_MAX_MEMORY_TOKENS", "768"))
DEFAULT_MAX_SUMMARY_TOKENS = int(os.getenv("REMEM_REASON_MAX_SUMMARY_TOKENS", "1024"))
DEFAULT_MAX_CHUNK_CHARS = int(os.getenv("REMEM_REASON_MAX_CHUNK_CHARS", "1600"))
DEFAULT_TEMPERATURE = float(os.getenv("REMEM_REASON_TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.getenv("REMEM_REASON_TOP_P", "1.0"))

NO_MEMORY = "No previous memory"
NO_RECALLED_MEMORY = "No memory was recalled."

TEMPLATE = """You are presented with a problem, a memory chunk that may contain useful evidence, a recalled memory, and a previous working memory.

Generate output in the following format:
- Output reasoning in <thinking>...</thinking>.
- Update the working memory in exactly one <update>...</update> block.
- If the current information is insufficient and you need to revisit earlier working memory, also output exactly one <recall>...</recall> block.

<problem>
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

<memory_chunk>
{chunk}
</memory_chunk>

Updated memory:
"""

TEMPLATE_FINAL = """You are assisting another agent. Based on the accumulated working memory, produce a concise reasoning context for a downstream tool-calling assistant.

Return exactly one <summary>...</summary> block containing:
- the key facts that matter for the current query
- relevant prior analyses / dataset context
- suggested next action for the downstream assistant

<problem>
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

Final reasoning context:
"""


@dataclass
class Candidate:
    candidate_id: str
    candidate_type: str
    text: str


def _tokenize(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return {token for token in cleaned.split() if token}


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - len("\n...(truncated)...\n")
    return text[:head] + "\n...(truncated)...\n" + text[-tail:]


def _parse_update_memory(text: str) -> str | None:
    match = re.search(r"<update>(.+?)</update>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"<recall>.*?</recall>", "", text, flags=re.DOTALL).strip()
    return cleaned or None


def _parse_recall_query(text: str) -> str | None:
    match = re.search(r"<recall>(.+?)</recall>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_summary(text: str) -> str | None:
    match = re.search(r"<summary>(.+?)</summary>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _retrieve_from_history(query: str, history_memories: list[str]) -> str | None:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return None
    best_score = 0.0
    best_memory = None
    for memory in history_memories:
        memory_tokens = _tokenize(memory)
        union = query_tokens | memory_tokens
        if not union:
            continue
        score = len(query_tokens & memory_tokens) / len(union)
        if score > best_score:
            best_score = score
            best_memory = memory
    return best_memory


async def _chat_once(
    session: ClientSession,
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    async with session.post(
        f"{UPSTREAM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {UPSTREAM_API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


async def healthz(_: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "upstream_base_url": UPSTREAM_BASE_URL,
            "default_model": DEFAULT_MODEL,
        }
    )


async def remem_reason(request: web.Request) -> web.Response:
    if SERVICE_API_TOKEN:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {SERVICE_API_TOKEN}":
            return web.json_response({"error": "unauthorized"}, status=401)

    payload = await request.json()
    query = str(payload.get("query", "") or "").strip()
    session_id = str(payload.get("session_id", "") or "")
    model = str(payload.get("model", "") or DEFAULT_MODEL).strip()
    if not query:
        return web.json_response({"error": "query is required"}, status=400)
    if not model:
        return web.json_response({"error": "model is required"}, status=400)

    config = payload.get("config", {}) or {}
    max_chunks = int(config.get("max_chunks", DEFAULT_MAX_CHUNKS))
    max_memory_tokens = int(config.get("max_memory_tokens", DEFAULT_MAX_MEMORY_TOKENS))
    max_summary_tokens = int(config.get("max_summary_tokens", DEFAULT_MAX_SUMMARY_TOKENS))
    max_chunk_chars = int(config.get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS))
    temperature = float(config.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(config.get("top_p", DEFAULT_TOP_P))

    raw_candidates = payload.get("candidates", []) or []
    candidates: list[Candidate] = []
    for i, item in enumerate(raw_candidates[:max_chunks]):
        text = _truncate(str(item.get("text", "") or ""), max_chunk_chars)
        if not text:
            continue
        candidates.append(
            Candidate(
                candidate_id=str(item.get("id", "") or f"candidate_{i}"),
                candidate_type=str(item.get("type", "") or "memory"),
                text=f"[MEMORY_TYPE={str(item.get('type', '') or 'memory')}]\n{text}",
            )
        )

    if not candidates:
        return web.json_response(
            {
                "summary": "",
                "used_memory_ids": [],
                "callbacks": [],
                "trace": {"session_id": session_id, "steps": 0},
            }
        )

    history_memories: list[str] = []
    working_memory = NO_MEMORY
    recalled_memory = NO_RECALLED_MEMORY
    used_memory_ids: list[str] = []
    callbacks: list[str] = []
    step_trace: list[dict[str, Any]] = []

    timeout = ClientTimeout(total=3600)
    async with ClientSession(timeout=timeout) as session:
        for idx, candidate in enumerate(candidates):
            prompt = TEMPLATE.format(
                prompt=query,
                recalled_memory=recalled_memory,
                memory=working_memory,
                chunk=candidate.text,
            )
            response = await _chat_once(
                session,
                prompt=prompt,
                model=model,
                max_tokens=max_memory_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            updated_memory = _parse_update_memory(response) or working_memory
            recall_query = _parse_recall_query(response)

            working_memory = updated_memory
            history_memories.append(updated_memory)
            used_memory_ids.append(candidate.candidate_id)

            if recall_query:
                callbacks.append(recall_query)
                recalled_memory = _retrieve_from_history(recall_query, history_memories) or NO_RECALLED_MEMORY
            else:
                recalled_memory = NO_RECALLED_MEMORY

            step_trace.append(
                {
                    "step": idx,
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": candidate.candidate_type,
                    "recall_query": recall_query,
                }
            )

        final_prompt = TEMPLATE_FINAL.format(
            prompt=query,
            recalled_memory=recalled_memory,
            memory=working_memory,
        )
        final_response = await _chat_once(
            session,
            prompt=final_prompt,
            model=model,
            max_tokens=max_summary_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    summary = _parse_summary(final_response) or final_response.strip()
    return web.json_response(
        {
            "summary": summary,
            "used_memory_ids": used_memory_ids,
            "callbacks": callbacks,
            "trace": {
                "session_id": session_id,
                "steps": len(step_trace),
                "step_trace": step_trace,
                "upstream_model": model,
            },
        }
    )


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/healthz", healthz)
    app.router.add_post("/remem_reason", remem_reason)
    return app


def main() -> None:
    app = create_app()
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
