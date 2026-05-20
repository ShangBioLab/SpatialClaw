"""
core.py — SpatialClaw Bot shared engine
=====================================
Platform-independent logic shared by Telegram and Feishu frontends:
LLM tool-use loop, skill execution, security helpers, audit logging.

Both frontends import this module, call ``init()`` once at startup, then
use the async helper functions to process user messages.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import requests
import shutil
import socket
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import traceback
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError
from spatialclaw.memory.layered_store import filter_context_preferences_for_continuation
from spatialclaw.memory.preference_policy import (
    preference_to_prompt_entry,
    validate_preference_payload,
)
from spatialclaw.memory.query_intent import (
    extract_memory_focus_terms,
    looks_like_memory_recall_query,
    looks_like_recent_or_output_query,
    looks_like_workflow_continuation_query,
)

# ---------------------------------------------------------------------------
# LLM provider presets  (Multi-Provider support)
# ---------------------------------------------------------------------------
# Each provider maps to (base_url, default_model, api_key_env_var).
# Users set LLM_PROVIDER=<key> for one-step configuration;
# LLM_BASE_URL and SPATIALCLAW_MODEL can still override.
#
# Inspired by EvoScientist's Multi-Provider architecture, adapted for
# SpatialClaw's lightweight AsyncOpenAI-based design. All providers are
# accessed through the OpenAI-compatible API protocol.

PROVIDER_PRESETS: dict[str, tuple[str, str, str]] = {
    # --- Tier 1: Primary providers ---
    "deepseek":   ("https://api.deepseek.com",                                    "deepseek-chat",          "DEEPSEEK_API_KEY"),
    "openai":     ("",                                                             "gpt-4o",                 "OPENAI_API_KEY"),
    "anthropic":  ("https://api.anthropic.com/v1/",                                "claude-sonnet-4-5-20250514", "ANTHROPIC_API_KEY"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/",     "gemini-2.5-flash",       "GOOGLE_API_KEY"),
    "nvidia":     ("https://integrate.api.nvidia.com/v1",                          "deepseek-ai/deepseek-r1", "NVIDIA_API_KEY"),

    # --- Tier 2: Third-party aggregators ---
    "siliconflow": ("https://api.siliconflow.cn/v1",                              "deepseek-ai/DeepSeek-V3", "SILICONFLOW_API_KEY"),
    "openrouter":  ("https://openrouter.ai/api/v1",                               "deepseek/deepseek-chat-v3-0324", "OPENROUTER_API_KEY"),
    "volcengine":  ("https://ark.cn-beijing.volces.com/api/v3",                   "doubao-1.5-pro-256k",     "VOLCENGINE_API_KEY"),
    "dashscope":   ("https://dashscope.aliyuncs.com/compatible-mode/v1",          "qwen-max",                "DASHSCOPE_API_KEY"),
    "zhipu":       ("https://open.bigmodel.cn/api/paas/v4",                       "glm-4-flash",             "ZHIPU_API_KEY"),

    # --- Tier 3: Local & custom ---
    "ollama":     ("http://localhost:11434/v1",                                    "qwen2.5:7b",             ""),
    "custom":     ("",                                                             "",                        ""),

}

# Ordered list for auto-detection: when LLM_PROVIDER is not set, we pick the
# first provider whose API key env var is present in the environment.
_PROVIDER_DETECT_ORDER = [
    "deepseek", "openai", "anthropic", "gemini", "nvidia",
    "siliconflow", "openrouter", "volcengine", "dashscope", "zhipu",
]


def resolve_provider(
    provider: str = "",
    base_url: str = "",
    model: str = "",
    api_key: str = "",
) -> tuple[str | None, str, str]:
    """Return (base_url_or_None, model, resolved_api_key) after applying provider defaults.

    Priority: explicit env vars > provider preset > auto-detect > hardcoded fallback.

    When *provider* is empty and *api_key* is empty, we scan provider-specific
    environment variables (DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, …)
    to auto-detect the provider.
    """
    provider_key = provider.lower().strip() if provider else ""

    # Auto-detect provider from available API keys
    if not provider_key and not api_key:
        for p in _PROVIDER_DETECT_ORDER:
            env_var = PROVIDER_PRESETS[p][2]
            if env_var and os.environ.get(env_var):
                provider_key = p
                api_key = os.environ[env_var]
                break

    # Look up preset
    preset = PROVIDER_PRESETS.get(provider_key, ("", "", ""))
    preset_url, preset_model, preset_key_env = preset

    # Allow per-provider base_url override via env var (e.g. ANTHROPIC_BASE_URL)
    env_base_url = ""
    if provider_key:
        env_base_url = os.environ.get(f"{provider_key.upper()}_BASE_URL", "")

    resolved_url = base_url or env_base_url or preset_url or None
    resolved_model = model or preset_model or "deepseek-chat"

    # Resolve API key: explicit > per-provider env > LLM_API_KEY fallback
    if not api_key and preset_key_env:
        api_key = os.environ.get(preset_key_env, "")
    if not api_key:
        api_key = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    return resolved_url, resolved_model, api_key


# ---------------------------------------------------------------------------
# Paths (relative to SpatialClaw project root)
# ---------------------------------------------------------------------------

from spatialclaw.paths import (
    DATA_DIR,
    EXAMPLES_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    ROOT_CLI as SPATIALCLAW_PY,
    SOUL_MD,
)
PYTHON = sys.executable

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PHOTO_BYTES = 20 * 1024 * 1024
MAX_DOWNLOAD_BYTES = int(os.getenv("SPATIALCLAW_MAX_DOWNLOAD_BYTES", str(50 * 1024 * 1024)))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from spatialclaw.core.registry import (
    build_skill_llm_cli_args,
    normalize_skill_llm_args,
    registry,
)
registry.load_all()

SPATIAL_DATA_EXTENSIONS = {
    f".{ext.lstrip('.')}"
    for domain in registry.domains.values()
    for ext in domain.get("primary_data_types", [])
    if ext != "*"
}
SPATIAL_DATA_EXTENSIONS.update({".csv", ".tsv", ".txt.gz"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("spatialclaw.bot")

# ---------------------------------------------------------------------------
# Skills table formatter (for /skills command in bot)
# ---------------------------------------------------------------------------


def format_skills_table(plain: bool = False) -> str:
    """Format all registered skills as categorized tables for bot display.

    Args:
        plain: If True, use ASCII markers instead of emoji (for platforms
               like Feishu where emoji gets stripped by strip_markup).
    """
    # Group skills by domain
    domain_skills: dict[str, list[tuple[str, dict]]] = {}
    for alias, info in registry.skills.items():
        d = info.get("domain", "other")
        domain_skills.setdefault(d, []).append((alias, info))

    total = len(registry.skills)
    if plain:
        lines = [f"SpatialClaw Skills ({total} total)", "=" * 40, ""]
    else:
        lines = [f"🔬 SpatialClaw Skills ({total} total)", ""]

    for domain_key, domain_info in registry.domains.items():
        skills_in_domain = domain_skills.get(domain_key, [])
        if not skills_in_domain:
            continue

        domain_name = domain_info.get("name", domain_key.title())
        data_types = domain_info.get("primary_data_types", [])
        types_str = ", ".join(f".{t}" if t != "*" else "*" for t in data_types)
        n = len(skills_in_domain)

        if plain:
            lines.append(f"[{domain_name}] ({n} skills, {types_str})")
            lines.append("~" * 40)
            for alias, info in skills_in_domain:
                script = info.get("script")
                tag = "[OK]" if script and script.exists() else "[--]"
                desc = info.get("description", "").split("—")[0].strip()
                lines.append(f"  {tag} {alias}")
                lines.append(f"       {desc}")
        else:
            lines.append(f"📂 {domain_name} [{types_str}]")
            for alias, info in skills_in_domain:
                script = info.get("script")
                status = "✅" if script and script.exists() else "📋"
                desc = info.get("description", "").split("—")[0].strip()
                lines.append(f"  {status} {alias}")
                lines.append(f"      {desc}")

        lines.append("")

    # Dynamically discovered skills not in known domains
    known = set(registry.domains.keys())
    extra = [(a, i) for a, i in registry.skills.items() if i.get("domain", "other") not in known]
    if extra:
        if plain:
            lines.append("[Other] (Dynamically Discovered)")
            lines.append("~" * 40)
        else:
            lines.append("📂 Other (Dynamically Discovered)")
        for alias, info in extra:
            script = info.get("script")
            desc = info.get("description", "").split("—")[0].strip()
            if plain:
                tag = "[OK]" if script and script.exists() else "[--]"
                lines.append(f"  {tag} {alias}")
                lines.append(f"       {desc}")
            else:
                status = "✅" if script and script.exists() else "📋"
                lines.append(f"  {status} {alias}")
                lines.append(f"      {desc}")
        lines.append("")

    if plain:
        lines.append("[OK] = ready  [--] = planned")
    else:
        lines.append("✅ = ready  📋 = planned")
    return "\n".join(lines)


def format_skill_extra_arg_help() -> str:
    """Build a positive, registry-derived CLI flag reference for LLM tools."""
    lines = ["Allowed skill-specific extra_args by canonical skill:"]
    for alias in sorted(registry.skills):
        info = registry.skills[alias]
        flags = sorted(info.get("allowed_extra_flags") or [])
        if not flags:
            continue
        mode_notes: list[str] = []
        if info.get("input_mode") == "directory":
            mode_notes.append("primary input is a sample directory")
        elif info.get("input_mode") == "file":
            mode_notes.append("primary input is a file")
        if info.get("alternative_input_flags"):
            mode_notes.append(
                "alternative input flags: "
                + ", ".join(sorted(info["alternative_input_flags"]))
            )
        if info.get("path_extra_flags"):
            mode_notes.append(
                "path-valued flags: "
                + ", ".join(sorted(info["path_extra_flags"]))
            )
        suffix = f" ({'; '.join(mode_notes)})" if mode_notes else ""
        lines.append(f"- {alias}: {', '.join(flags)}{suffix}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audit log (JSONL)
# ---------------------------------------------------------------------------

_AUDIT_LOG_DIR = PROJECT_ROOT / "bot" / "logs"
_AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
_AUDIT_LOG_PATH = _AUDIT_LOG_DIR / "audit.jsonl"


def audit(event: str, **kwargs):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
    try:
        with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.warning(f"Audit log write failed: {e}")


# ---------------------------------------------------------------------------
# Module-level state (initialised by init())
# ---------------------------------------------------------------------------

llm: AsyncOpenAI | None = None
SPATIALCLAW_MODEL: str = "deepseek-chat"
LLM_PROVIDER_NAME: str = ""

conversations: dict[int | str, list] = {}
_conversation_access: dict[int | str, float] = {}  # LRU tracking
MAX_HISTORY = int(os.getenv("SPATIALCLAW_MAX_HISTORY", "50"))
MAX_CONVERSATIONS = int(os.getenv("SPATIALCLAW_MAX_CONVERSATIONS", "1000"))

received_files: dict[int | str, dict] = {}
pending_media: dict[int | str, list[dict]] = {}
pending_text: list[str] = []

BOT_START_TIME = time.time()

# Memory system (optional)
memory_store = None
session_manager = None
remem_operator = None

# ---------------------------------------------------------------------------
# Usage statistics (token counters)
# ---------------------------------------------------------------------------

_usage: dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "api_calls": 0,
}

# Approximate pricing per 1M tokens (USD) — keyed by provider:model fragment.
# These are reference values; override via LLM_INPUT_PRICE / LLM_OUTPUT_PRICE env vars.
_TOKEN_PRICES: dict[str, tuple[float, float]] = {
    # (input $/1M, output $/1M)
    "deepseek-chat":        (0.27,  1.10),
    "deepseek-reasoner":    (0.55,  2.19),
    "gpt-4o":               (2.50, 10.00),
    "gpt-4o-mini":          (0.15,  0.60),
    "gpt-4-turbo":          (10.0, 30.00),
    "gpt-3.5-turbo":        (0.50,  1.50),
    "claude-3-5-sonnet":    (3.00, 15.00),
    "claude-3-5-haiku":     (0.80,  4.00),
    "claude-3-opus":        (15.0, 75.00),
    "gemini-1.5-pro":       (1.25,  5.00),
    "gemini-1.5-flash":     (0.075, 0.30),
    "gemini-2.0-flash":     (0.10,  0.40),
    "qwen-plus":            (0.40,  1.20),
    "qwen-long":            (0.05,  0.20),
}


def _get_token_price(model: str) -> tuple[float, float]:
    """Return (input_price, output_price) per 1M tokens for the current model."""
    # Allow explicit override via env vars
    try:
        inp = float(os.environ.get("LLM_INPUT_PRICE", ""))
        out = float(os.environ.get("LLM_OUTPUT_PRICE", ""))
        return inp, out
    except (ValueError, TypeError):
        pass
    model_lower = model.lower()
    for key, prices in _TOKEN_PRICES.items():
        if key in model_lower:
            return prices
    return (0.0, 0.0)  # Unknown model — no cost estimate


def _accumulate_usage(response_usage) -> dict[str, int]:
    """Add API response usage to global counters. Returns per-call delta."""
    if response_usage is None:
        return {}

    def _usage_value(obj, *names: str) -> int:
        for name in names:
            if isinstance(obj, dict):
                value = obj.get(name)
            else:
                value = getattr(obj, name, None)
            if value is not None:
                return value or 0
        return 0

    delta = {
        "prompt_tokens":     _usage_value(response_usage, "prompt_tokens", "input_tokens"),
        "completion_tokens": _usage_value(response_usage, "completion_tokens", "output_tokens"),
        "total_tokens":      _usage_value(response_usage, "total_tokens"),
    }
    _usage["prompt_tokens"]     += delta["prompt_tokens"]
    _usage["completion_tokens"] += delta["completion_tokens"]
    _usage["total_tokens"]      += delta["total_tokens"]
    _usage["api_calls"]         += 1
    return delta


def _extract_message_text(content) -> str:
    """Flatten message content blocks into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type in ("text", "output_text"):
                text = block.get("text", "")
                if isinstance(text, dict):
                    text = text.get("value", "")
                if text:
                    texts.append(str(text))
        return "\n".join(texts).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return str(content)


def _normalise_tool_calls(tool_calls) -> list[dict[str, str]]:
    """Convert SDK/dict tool-call objects into a simple internal structure."""
    normalised: list[dict[str, str]] = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            function = tc.get("function") or {}
            tc_id = tc.get("id", "")
            name = function.get("name", "")
            arguments = function.get("arguments", "{}")
        else:
            function = getattr(tc, "function", None)
            tc_id = getattr(tc, "id", "")
            name = getattr(function, "name", "") if function else ""
            arguments = getattr(function, "arguments", "{}") if function else "{}"
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalised.append({
            "id": tc_id or "",
            "name": name or "",
            "arguments": arguments if isinstance(arguments, str) else str(arguments),
        })
    return normalised


def _extract_responses_output(output_items) -> tuple[str, list[dict[str, str]]]:
    """Best-effort extraction for gateways that return Responses-style payloads."""
    texts: list[str] = []
    tool_calls: list[dict[str, str]] = []

    for item in output_items or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        if item_type == "message":
            texts.append(_extract_message_text(item.get("content")))
            continue
        if item_type in ("text", "output_text"):
            texts.append(_extract_message_text(item))
            continue
        if item_type in ("function_call", "tool_call"):
            arguments = item.get("arguments", item.get("input", "{}"))
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or "",
                "name": item.get("name", ""),
                "arguments": arguments if isinstance(arguments, str) else str(arguments),
            })

    text = "\n".join(part for part in texts if part).strip()
    return text, tool_calls


def _normalise_chat_response(response) -> dict[str, object]:
    """Normalise provider responses into {usage, content, tool_calls}."""
    original_response = response

    if isinstance(response, str):
        stripped = response.strip()
        if not stripped:
            return {"usage": None, "content": "", "tool_calls": []}
        try:
            response = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("LLM endpoint returned plain text instead of a structured completion object")
            return {"usage": None, "content": response, "tool_calls": []}

    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return {
                "usage": response.get("usage"),
                "content": _extract_message_text(message.get("content")),
                "tool_calls": _normalise_tool_calls(message.get("tool_calls")),
            }

        output = response.get("output")
        if output is not None:
            content, tool_calls = _extract_responses_output(output)
            return {
                "usage": response.get("usage"),
                "content": content,
                "tool_calls": tool_calls,
            }

        raise TypeError(
            "LLM endpoint returned JSON without 'choices' or 'output'. "
            "This project expects an OpenAI-compatible chat completion payload. "
            "Check LLM_BASE_URL; most compatible gateways use a /v1 base URL."
        )

    choices = getattr(response, "choices", None)
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is None:
            raise TypeError("LLM endpoint returned a completion without a message object.")
        return {
            "usage": getattr(response, "usage", None),
            "content": _extract_message_text(getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")),
            "tool_calls": _normalise_tool_calls(getattr(message, "tool_calls", None) if not isinstance(message, dict) else message.get("tool_calls")),
        }

    raise TypeError(
        f"Unsupported LLM response type: {type(original_response).__name__}. "
        "Check LLM_BASE_URL and gateway compatibility with OpenAI chat/completions."
    )


def get_usage_snapshot() -> dict:
    """Return a copy of the current cumulative usage statistics plus cost estimate."""
    inp_price, out_price = _get_token_price(SPATIALCLAW_MODEL)
    cost = (
        _usage["prompt_tokens"]     / 1_000_000 * inp_price +
        _usage["completion_tokens"] / 1_000_000 * out_price
    )
    return {
        **_usage,
        "model": SPATIALCLAW_MODEL,
        "provider": LLM_PROVIDER_NAME,
        "input_price_per_1m":  inp_price,
        "output_price_per_1m": out_price,
        "estimated_cost_usd":  round(cost, 6),
    }


def reset_usage() -> None:
    """Reset session-level usage counters to zero."""
    for k in _usage:
        _usage[k] = 0



# ---------------------------------------------------------------------------
# Shared rate limiter (used by both Telegram and Feishu)
# ---------------------------------------------------------------------------

RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "10"))
_rate_buckets: dict[str, list[float]] = {}


def check_rate_limit(user_id: str, admin_id: str = "") -> bool:
    """Check per-user rate limit. Returns True if allowed."""
    if RATE_LIMIT_PER_HOUR <= 0 or (admin_id and user_id == admin_id):
        return True
    now = time.time()
    bucket = _rate_buckets.setdefault(user_id, [])
    bucket[:] = [t for t in bucket if now - t < 3600]
    if len(bucket) >= RATE_LIMIT_PER_HOUR:
        return False
    bucket.append(now)
    return True


def _evict_lru_conversations():
    """Evict least-recently-used conversations when limit exceeded."""
    if len(conversations) <= MAX_CONVERSATIONS:
        return
    # Sort by access time, evict oldest
    sorted_keys = sorted(_conversation_access, key=_conversation_access.get)
    to_evict = len(conversations) - MAX_CONVERSATIONS
    for key in sorted_keys[:to_evict]:
        conversations.pop(key, None)
        _conversation_access.pop(key, None)
    logger.debug(f"Evicted {to_evict} stale conversation(s)")


# ---------------------------------------------------------------------------
# Memory Auto-Capture Helpers
# ---------------------------------------------------------------------------

def _infer_dataset_metadata(input_path: str | Path, data_type: str = "") -> dict[str, object | None]:
    """Best-effort dataset metadata extraction for memory auto-capture.

    The memory layer stores relative paths only, so this helper focuses on
    safe, reusable metadata: platform plus dataset shape. It avoids loading
    full matrices when a lightweight backed/metadata-only path is available.
    """
    path = Path(input_path)
    metadata: dict[str, object | None] = {
        "platform": None,
        "n_obs": None,
        "n_vars": None,
    }

    explicit_platform = str(data_type or "").strip()
    if explicit_platform and explicit_platform.lower() != "auto":
        metadata["platform"] = explicit_platform

    def _update_shape(n_obs: object, n_vars: object) -> None:
        if n_obs is not None and n_vars is not None:
            try:
                metadata["n_obs"] = int(n_obs)
                metadata["n_vars"] = int(n_vars)
            except (TypeError, ValueError):
                pass

    def _read_h5ad_shape(h5ad_path: Path) -> None:
        try:
            import anndata as ad
        except Exception:
            return

        adata = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adata = ad.read_h5ad(h5ad_path, backed="r")
            _update_shape(*adata.shape)
            if metadata["platform"] is None:
                obsm_keys = set(adata.obsm.keys())
                if "spatial" in obsm_keys or "spatial" in getattr(adata, "uns", {}):
                    metadata["platform"] = "Spatial"
        except Exception:
            return
        finally:
            try:
                if adata is not None and getattr(adata, "file", None) is not None:
                    adata.file.close()
            except Exception:
                pass

    def _read_10x_h5_shape(h5_path: Path) -> None:
        try:
            import h5py
        except Exception:
            return

        try:
            with h5py.File(h5_path, "r") as h5:
                if "matrix" in h5 and "shape" in h5["matrix"]:
                    shape = h5["matrix"]["shape"][()]
                    if len(shape) == 2:
                        n_vars, n_obs = shape
                        _update_shape(n_obs, n_vars)
                elif "X" in h5 and hasattr(h5["X"], "shape"):
                    _update_shape(*h5["X"].shape)
        except Exception:
            return

    if path.is_dir():
        visium_h5 = sorted(path.glob("*_feature_bc_matrix.h5"))
        if visium_h5:
            if metadata["platform"] is None and (path / "spatial").is_dir():
                metadata["platform"] = "Visium"
            _read_10x_h5_shape(visium_h5[0])
            return metadata

        h5ad_candidates = sorted(path.glob("*.h5ad"))
        if h5ad_candidates:
            _read_h5ad_shape(h5ad_candidates[0])
        return metadata

    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        _read_h5ad_shape(path)
    elif suffix == ".h5":
        _read_10x_h5_shape(path)

    return metadata


def _memory_safe_path(path_value: str | Path | None) -> str:
    """Return a readable path for memory prompts without leaking external absolutes."""
    if not path_value:
        return ""

    path = Path(path_value)
    if not path.is_absolute():
        return str(path)

    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path

    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        if resolved.is_absolute():
            return resolved.name
        return str(path)


async def _auto_capture_dataset(session_id: str, input_path: str, data_type: str = ""):
    """Auto-capture dataset memory when a file is processed."""
    if not memory_store or not session_id or not input_path:
        return

    try:
        from spatialclaw.memory.layered_store import DatasetMemory

        rel_path = _memory_safe_path(input_path)

        metadata = _infer_dataset_metadata(input_path, data_type=data_type)

        ds_mem = DatasetMemory(
            file_path=rel_path,
            platform=metadata["platform"],
            n_obs=metadata["n_obs"],
            n_vars=metadata["n_vars"],
            preprocessing_state="raw",
        )
        await memory_store.save_memory(session_id, ds_mem)
        logger.debug(f"Auto-captured dataset: {rel_path}")
    except Exception as e:
        logger.warning(f"Auto-capture dataset failed: {e}")

def _read_report_as_trajectory(output_dir, skill: str, success: bool) -> str:
    base = f"> {skill}\nstatus: {'completed' if success else 'failed'}\n"
    if not output_dir:
        return base
    try:
        output_path = Path(output_dir)
        # 按优先级查找报告文件
        candidates = (
            list(output_path.glob("report.md")) +
            list(output_path.glob("*.md")) +
            list(output_path.glob("summary.txt")) +
            list(output_path.glob("*results*.csv")) +
            list(output_path.rglob("*.txt"))  # 包含子目录的 txt
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                content = candidate.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not content.strip():
                continue

            if candidate.suffix == ".csv":
                # CSV 只取前 20 行
                lines = content.split("\n")[:20]
                snippet = "\n".join(lines)
            elif candidate.suffix in (".txt", ".log"):
                # txt/log 取前 50 行
                lines = content.split("\n")[:50]
                snippet = "\n".join(lines)
            else:
                # md 过滤掉 Methods/图片
                lines = []
                skip = False
                for line in content.split("\n"):
                    if line.startswith("## Methods") or \
                       line.startswith("## Reproducibility"):
                        skip = True
                    elif line.startswith("## "):
                        skip = False
                    if not skip and not line.startswith("!["):
                        lines.append(line)
                snippet = "\n".join(lines)

            if len(snippet.strip()) > 50:  # 有实质内容才用
                return base + snippet[:2000]

    except Exception:
        pass
    return base

async def _auto_capture_analysis(
    session_id: str, skill: str, args: dict, output_dir: Path, success: bool
):
    if not memory_store or not session_id:
        return
    try:
        from spatialclaw.memory.layered_store import AnalysisMemory
        from spatialclaw.memory.insight_extractor import InsightExtractor

        method = args.get("method", "default")
        input_path = args.get("file_path", "")
        file_name = Path(input_path).stem if input_path else ""
        safe_input_path = _memory_safe_path(input_path)
        safe_output_path = _memory_safe_path(output_dir)

        task_main = f"{skill} on {file_name}" if file_name else skill
        if method and method != "default":
            task_main = f"{method} {task_main}"

        source_dataset_id = ""
        try:
            datasets = await memory_store.get_memories(
                session_id, "dataset", limit=1, layer="episodic"
            )
            if datasets:
                source_dataset_id = datasets[0].memory_id
        except Exception:
            pass

        memory = AnalysisMemory(
            source_dataset_id=source_dataset_id or "",
            skill=skill,
            method=method,
            parameters={"input": safe_input_path} if safe_input_path else {},
            output_path=safe_output_path,
            status="completed" if success else "failed",
            task_main=task_main,
            task_trajectory=_read_report_as_trajectory(output_dir, skill, success),
            key_steps=f"Run {skill} with method={method}",
            fail_reason="" if success else f"{skill} execution failed",
        )

        # Save through the typed layered store so promotion policy owns placement.
        await memory_store.save_memory(session_id, memory)
        # Assign a coarse cluster ID by spatial skill family.
        skill_cluster_map = {
            "spatial-enrichment":  0,
            "spatial-preprocessing":  1,
            "spatial-domain-identification": 2,
            "spatial-deconvolution": 3,
            "spatial-de":          4,
            "spatial-cell-annotation": 7,
        }
        cluster_id = skill_cluster_map.get(skill, 99)
        try:
            await memory_store.update_memory(memory.memory_id, {"cluster_id": cluster_id})
        except Exception:
            pass
        
        # Link related analysis memories through their real layered graph URIs.
        try:
            source_uris = await memory_store.get_memory_uris(session_id, memory)
            source_uri = source_uris[0] if source_uris else ""

            def _jaccard(a: str, b: str) -> float:
                sa, sb = set(a.lower().split()), set(b.lower().split())
                return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

            if not source_uri:
                raise ValueError("Saved analysis URI was not found")

            candidates = await memory_store.get_memories(
                session_id, "analysis", limit=100
            )
            for candidate in candidates:
                if candidate.memory_id == memory.memory_id:
                    continue
                if not candidate.task_main:
                    continue
                candidate_uris = await memory_store.get_memory_uris(
                    session_id, candidate
                )
                candidate_uri = candidate_uris[0] if candidate_uris else ""
                if not candidate_uri:
                    continue
                sim = _jaccard(memory.task_main, candidate.task_main)
                if sim >= 0.2:
                    await memory_store._client._graph.add_task_similarity_edge(
                        source_uri, candidate_uri, sim
                    )
        except Exception as edge_err:
            logger.debug(f"Task similarity edge skipped: {edge_err}")
            
        logger.debug(f"Auto-captured analysis: {skill} ({method})")

        try:
            def _insight_llm(system: str, user: str) -> str:
                from openai import OpenAI
                base_url, _, api_key = resolve_provider()
                client = OpenAI(
                    api_key=api_key,
                    **({"base_url": base_url} if base_url else {})
                )
                resp = client.chat.completions.create(
                    model=SPATIALCLAW_MODEL,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return resp.choices[0].message.content or ""

            extractor = InsightExtractor(
                store=memory_store,
                llm_callable=_insight_llm,
            )
            all_analyses = await memory_store.get_memories(
                session_id, "analysis", limit=500
            )
            await extractor.on_task_complete(
                session_id=session_id,
                analysis=memory,
                memory_size=len(all_analyses),
            )
        except Exception as insight_err:
            logger.debug(f"Insight extraction skipped: {insight_err}")

    except Exception as _err:
        logger.warning(f"Auto-capture analysis failed: {_err}")

# ---------------------------------------------------------------------------
# Insight Memory Helpers
# ---------------------------------------------------------------------------

_INSIGHT_TYPE_ALIASES = {
    "cluster": "cluster",
    "clusters": "cluster",
    "domain": "spatial_domain",
    "domains": "spatial_domain",
    "spatial_domain": "spatial_domain",
    "spatial-domain": "spatial_domain",
    "spatial domain": "spatial_domain",
    "cell_type": "cell_type",
    "cell-type": "cell_type",
    "cell type": "cell_type",
}


def _normalize_insight_entity_type(entity_type: str) -> str:
    raw = str(entity_type or "").strip().lower()
    if not raw:
        return "cluster"
    raw = re.sub(r"[\s-]+", "_", raw)
    return _INSIGHT_TYPE_ALIASES.get(raw, raw)


def _canonicalize_insight_identity(entity_type: str, entity_id: str) -> tuple[str, str]:
    """Normalize noisy LLM insight identities into stable memory keys."""
    normalized_type = _normalize_insight_entity_type(entity_type)
    raw_id = str(entity_id or "").strip().lower()
    raw_id = re.sub(r"[\s-]+", "_", raw_id)
    raw_id = re.sub(r"_+", "_", raw_id).strip("_")

    if not raw_id:
        return normalized_type, raw_id

    cluster_match = re.fullmatch(r"cluster_?(.+)", raw_id)
    if cluster_match:
        suffix = cluster_match.group(1).strip("_")
        return "cluster", f"cluster_{suffix}" if suffix else "cluster"

    domain_match = re.fullmatch(r"(?:spatial_)?domain_?(.+)", raw_id)
    if domain_match:
        suffix = domain_match.group(1).strip("_")
        return "spatial_domain", f"domain_{suffix}" if suffix else "domain"

    if raw_id.isdigit():
        if normalized_type == "spatial_domain":
            return normalized_type, f"domain_{raw_id}"
        if normalized_type == "cluster":
            return normalized_type, f"cluster_{raw_id}"

    return normalized_type, raw_id


def _insight_numeric_slot(entity_type: str, entity_id: str) -> str | None:
    normalized_type, normalized_id = _canonicalize_insight_identity(entity_type, entity_id)
    if normalized_type not in {"cluster", "spatial_domain"}:
        return None
    match = re.search(r"(\d+)$", normalized_id)
    return match.group(1) if match else None


async def _get_latest_analysis_memory(session_id: str):
    if not memory_store or not session_id:
        return None
    analyses = await memory_store.get_memories(
        session_id,
        "analysis",
        limit=1,
        layer="all",
    )
    return analyses[0] if analyses else None


async def _find_existing_insight_memory(session_id: str, entity_type: str, entity_id: str):
    if not memory_store or not session_id:
        return None

    requested_type, requested_id = _canonicalize_insight_identity(entity_type, entity_id)
    requested_slot = _insight_numeric_slot(requested_type, requested_id)
    insights = await memory_store.get_memories(
        session_id,
        "insight",
        limit=50,
        layer="all",
    )

    for insight in insights:
        existing_type, existing_id = _canonicalize_insight_identity(
            getattr(insight, "entity_type", ""),
            getattr(insight, "entity_id", ""),
        )
        if existing_type == requested_type and existing_id == requested_id:
            return insight

    if requested_slot is None:
        return None

    for insight in insights:
        existing_type = _normalize_insight_entity_type(getattr(insight, "entity_type", ""))
        existing_slot = _insight_numeric_slot(
            getattr(insight, "entity_type", ""),
            getattr(insight, "entity_id", ""),
        )
        if existing_slot == requested_slot and existing_type in {"cluster", "spatial_domain"}:
            return insight

    return None


def _format_memory_focus_item(memory) -> str:
    memory_type = getattr(memory, "memory_type", "")
    if memory_type == "dataset":
        return (
            f"- Dataset: {_memory_safe_path(memory.file_path)} "
            f"({memory.platform or 'unknown'}, {memory.n_obs or '?'} obs, {memory.preprocessing_state})"
        )
    if memory_type == "analysis":
        line = f"- Analysis: {memory.skill} ({memory.method}) - {memory.status}"
        if getattr(memory, "output_path", ""):
            line += f" | output: {_memory_safe_path(memory.output_path)}"
        return line
    if memory_type == "insight":
        confidence = "confirmed" if memory.confidence == "user_confirmed" else "predicted"
        line = f"- Insight: {memory.entity_type} {memory.entity_id} -> {memory.biological_label} ({confidence})"
        if getattr(memory, "source_analysis_id", ""):
            line += f" | analysis_id: {memory.source_analysis_id}"
        return line
    if memory_type == "preference":
        prompt_entry = preference_to_prompt_entry(memory.key, memory.value)
        if prompt_entry is None:
            return "- Preference: [hidden unsafe preference payload]"
        prompt_key, prompt_value = prompt_entry
        return f"- Preference: {prompt_key} = {prompt_value}"
    return f"- Memory: {memory_type}"


async def _build_targeted_memory_context(session_id: str, user_text: str) -> str:
    if (
        not memory_store
        or not session_id
        or not (
            looks_like_memory_recall_query(user_text)
            or looks_like_workflow_continuation_query(user_text)
        )
    ):
        return ""

    parts = [
        "## Relevant Memory For This Question",
        "Use the records below as the primary source of truth for the current memory question.",
    ]

    continuation_query = looks_like_workflow_continuation_query(user_text)
    if (
        continuation_query
        or looks_like_recent_or_output_query(user_text)
    ):
        datasets = await memory_store.get_memories(session_id, "dataset", limit=1, layer="all")
        analyses = await memory_store.get_memories(session_id, "analysis", limit=1, layer="all")
        if datasets:
            parts.append("**Most Recent Dataset**:")
            parts.append(_format_memory_focus_item(datasets[0]))
        if analyses:
            parts.append("**Most Recent Analysis**:")
            parts.append(_format_memory_focus_item(analyses[0]))
            if continuation_query and getattr(analyses[0], "status", "") == "completed":
                resolution_lines = [
                    "**Workflow Continuation Resolution**:",
                    "When the user asks to continue or resume the previous workflow, use the most recent completed analysis as the source of truth for workflow state.",
                    "Do not let saved method preferences override the method, parameters, or output path of that completed workflow unless the user explicitly asks to switch methods.",
                    f"- Resolved workflow: {analyses[0].skill}",
                    f"- Resolved method: {analyses[0].method}",
                ]
                if getattr(analyses[0], "output_path", ""):
                    resolution_lines.append(f"- Resolved output path: {_memory_safe_path(analyses[0].output_path)}")
                if datasets:
                    resolution_lines.append(f"- Resolved dataset: {_memory_safe_path(datasets[0].file_path)}")
                if getattr(analyses[0], "parameters", None):
                    safe_parameters = dict(analyses[0].parameters)
                    for key in ("input", "input_path", "output", "output_path"):
                        if key in safe_parameters:
                            safe_parameters[key] = _memory_safe_path(safe_parameters[key])
                    resolution_lines.append(
                        f"- Resolved parameters: {json.dumps(safe_parameters, ensure_ascii=False, sort_keys=True)}"
                    )
                parts.extend(resolution_lines)

    matches = []
    seen_memory_ids: set[str] = set()
    for term in extract_memory_focus_terms(user_text):
        for memory in await memory_store.search_memories(session_id, term, layer="all"):
            if memory.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory.memory_id)
            matches.append(memory)

    if matches:
        parts.append("**Direct Matches**:")
        for memory in matches[:5]:
            parts.append(_format_memory_focus_item(memory))

    return "\n\n".join(parts) if len(parts) > 2 else ""


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages user sessions with memory persistence."""

    def __init__(self, store):
        self.store = store

    async def get_or_create(self, user_id: str, platform: str, chat_id: str):
        """Get existing session or create new one."""
        session_id = f"{platform}:{user_id}:{chat_id}"
        session = await self.store.get_session(session_id)
        if not session:
            session = await self.store.create_session(user_id, platform, chat_id, session_id=session_id)
        else:
            await self.store.update_session(session_id, {"last_activity": datetime.now(timezone.utc)})
        return session

    async def load_context(self, session_id: str, user_text: str = "") -> str:
        try:
            base_context = await self.store.load_context(session_id)
        except Exception as e:
            logger.error(f"Failed to load memory context: {e}")
            return ""

        if looks_like_workflow_continuation_query(user_text):
            base_context = filter_context_preferences_for_continuation(
                base_context
            )

        # Inject related historical tasks and reusable insight rules.
        if not user_text:
            return base_context

        try:
            from spatialclaw.memory.insight_extractor import InsightExtractor

            def _noop_llm(s, u):
                return ""

            extractor = InsightExtractor(store=self.store, llm_callable=_noop_llm)
            successful, failed, insights = await extractor.retrieve_for_task(
                session_id=session_id,
                task_main=user_text[:200],
                successful_topk=2,
                failed_topk=1,
                insight_topk=3,
            )

            memory_parts = []
            if successful:
                memory_parts.append("## 相似成功案例")
                for s in successful:
                    memory_parts.append(f"- {s.task_main} | 方法: {s.method}")
                    if s.key_steps:
                        memory_parts.append(f"  步骤: {s.key_steps}")
            if failed:
                memory_parts.append("## 相似失败案例")
                for f in failed:
                    memory_parts.append(f"- {f.task_main}")
                    if f.fail_reason:
                        memory_parts.append(f"  原因: {f.fail_reason}")
            if insights:
                memory_parts.append("## 经验规则")
                for ins in insights:
                    memory_parts.append(f"- {ins.rule}")

            if memory_parts:
                memory_text = "\n".join(memory_parts)
                return f"{memory_text}\n\n{base_context}" if base_context else memory_text

        except Exception as e:
            logger.debug(f"Layered memory retrieval skipped: {e}")

        return base_context


def init(
    api_key: str = "",
    base_url: str | None = None,
    model: str = "",
    provider: str = "",
):
    """Initialise the shared LLM client. Call once at startup.

    ``provider`` selects a preset (deepseek, gemini, openai, anthropic,
    nvidia, siliconflow, openrouter, volcengine, dashscope, zhipu, ollama,
    custom).  Explicit ``base_url`` / ``model`` override the preset.

    When ``api_key`` is empty, the key is auto-resolved from provider-
    specific environment variables (e.g. DEEPSEEK_API_KEY for deepseek).
    """
    global llm, SPATIALCLAW_MODEL, LLM_PROVIDER_NAME, memory_store, session_manager, remem_operator

    resolved_url, resolved_model, resolved_key = resolve_provider(
        provider=provider,
        base_url=base_url or "",
        model=model,
        api_key=api_key,
    )
    SPATIALCLAW_MODEL = resolved_model

    # Determine display name for the provider
    if provider:
        LLM_PROVIDER_NAME = provider
    elif resolved_url:
        # Try to match resolved_url back to a known provider
        for pname, (purl, _, _) in PROVIDER_PRESETS.items():
            if purl and resolved_url and purl.rstrip("/") in resolved_url.rstrip("/"):
                LLM_PROVIDER_NAME = pname
                break
        else:
            LLM_PROVIDER_NAME = "custom"
    else:
        LLM_PROVIDER_NAME = "openai"

    kw: dict = {"api_key": resolved_key or api_key}
    if resolved_url:
        kw["base_url"] = resolved_url
    llm = AsyncOpenAI(**kw)

    logger.info(
        f"LLM initialised: provider={LLM_PROVIDER_NAME}, "
        f"model={SPATIALCLAW_MODEL}, base_url={resolved_url or '(default)'}"
    )

    # Memory initialization — uses the new graph-based memory system
    # Enabled by default; disable with SPATIALCLAW_MEMORY_ENABLED=false
    if os.getenv("SPATIALCLAW_MEMORY_ENABLED", "true").lower() not in ("false", "0", "no"):
        try:
            from spatialclaw.memory.layered_store import LayeredMemoryStore

            db_url = os.getenv("SPATIALCLAW_MEMORY_DB_URL")  # None = use default ./.config/spatialclaw/memory.db

            store = LayeredMemoryStore(
                database_url=db_url,
            )
            # NOTE: initialize() is called lazily on first async operation
            # from MemoryClient._ensure_init(), since init() runs in sync context.

            memory_store = store
            session_manager = SessionManager(store)
            logger.info("Graph memory system initialized (spatialclaw.memory)")
        except ImportError:
            logger.warning("Memory dependencies not installed, skipping memory init")
        except Exception as e:
            logger.error(f"Memory init failed: {e}")

    mar_enabled = os.getenv("SPATIALCLAW_MAR_ENABLED", "").lower()
    remem_enabled = os.getenv("SPATIALCLAW_REMEM_ENABLED", "false").lower()
    if mar_enabled not in ("", "false", "0", "no") or remem_enabled not in ("false", "0", "no"):
        try:
            from spatialclaw.agents.mar_operator import MemoryAugmentedReasoningOperator

            remem_operator = MemoryAugmentedReasoningOperator.from_env()
            if remem_operator is None:
                logger.warning(
                    "MAR/ReMemR1 operator enabled but not configured. "
                    "Set SPATIALCLAW_MAR_BASE_URL and SPATIALCLAW_MAR_MODEL, "
                    "or keep using the SPATIALCLAW_REMEM_* fallback variables."
                )
            else:
                logger.info(
                    "MAR/ReMemR1 operator initialized: model=%s base_url=%s",
                    remem_operator.model,
                    remem_operator.base_url,
                )
        except Exception as e:
            remem_operator = None
            logger.error(f"MAR/ReMemR1 operator init failed: {e}")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def get_role_guardrails() -> str:
    domain_names = ", ".join(d["name"].lower() for d in registry.domains.values())

    routing_lines = []
    for alias, info in registry.skills.items():
        if info.get("interface") == "python-api":
            continue
        desc = info.get("description", alias).split("—")[0].split("(")[0].strip()
        routing_lines.append(f"   - {desc}: skill='{alias}'")

    routing_text = "\n".join(routing_lines)

    return f"""
Operational constraints:
1. You are a spatial transcriptomics analysis assistant powered by SpatialClaw skills.
2. Supported domains: {domain_names}.
3. Keep outputs concise, evidence-led, and explicit about confidence and gaps.
4. When the user sends spatial data or asks about analysis, use the spatialclaw tool. Skill routing:
{routing_text}
   - Auto-detect skill: skill='auto'
4b. METHOD PARAMETER (CRITICAL): When user specifies a method, pass it LOWERCASE via method parameter:
   - spatial-deconvolution methods: tangram, stereoscope, graphst
   - spatial-domain-identification methods: leiden, louvain, spagcn, stagate, graphst
   - spatial-cell-annotation methods: tangram, scanvi, cellassign, sctype
   - spatial-cell-communication methods: liana, cellphonedb, fastccc
   - spatial-trajectory methods: dpt, cellrank, palantir
   - spatial-svg-detection methods: spatialde, flashs
   - spatial-integration methods: harmony, bbknn, scanorama
   - spatial-omics-integrate methods: spatialglue, spaddm
   - spatial-velocity methods: scvelo, velovi
   Examples: "GraphST" → method='graphst', "Tangram" → method='tangram'
   IMPORTANT: Deep learning methods (stereoscope, tangram, graphst, spagcn, stagate, spatialglue, spaddm, scvi, velovi)
   may take 10-30 minutes. Inform user: "This will take 10-30 minutes, please wait..."
5. TOOL OUTPUT RELAY (STRICT): When the spatialclaw tool returns results, relay
   the output VERBATIM. Do not paraphrase, summarise, or rewrite. The output
   contains precise numerical data that must not be altered. You may add a brief
   intro line but never replace or condense the tool output.
6. For uploaded data or visual evidence, suggest appropriate SpatialClaw analysis skills.
7. For quick demos: say "run spatial-preprocessing demo", "run spatial-de demo", etc.
   Use mode='demo' to run with built-in synthetic data.
8. FILE PATH MODE (IMPORTANT): Spatial data files are often too large to upload
   via messaging. When the user mentions a file path or filename, use mode='path'
   and set file_path to the path or filename they provided. The system will automatically search
   trusted directories.
   **CRITICAL FILE USAGE RULES**:
   - When the user specifies a file path, use EXACTLY that file for the requested operation.
   - Run preprocessing or other preparatory steps only when the user explicitly asks.
   - Treat the specified path as authoritative.
   - If the operation fails because the file needs preprocessing, report the failure and ask the user how to proceed.
   Examples:
   - User: "分析 data/brain_visium.h5ad" → mode='path', file_path='data/brain_visium.h5ad'
   - User: "run spatial-preprocessing on my_data.h5ad" → mode='path', file_path='my_data.h5ad'
   - User: "对 /mnt/nas/brain_visium.h5ad 做质量控制" → mode='path', file_path='/mnt/nas/brain_visium.h5ad', skill='spatial-preprocessing'
   - User: "做GO富集分析，groupby是leiden" → skill='spatial-enrichment', file_path='data/xxx.h5ad', extra_args=['--analysis-type','go','--groupby','leiden','--organism','human']\n"
9. NO CODE GENERATION (STRICT): You are an analysis assistant, NOT a code generator.
   - NEVER proactively create Python scripts, shell scripts, or code files.
   - NEVER use write_file to generate .py, .sh, .r, .R, or other script files.
   - All analysis MUST go through the spatialclaw tool — the skills already implement
     the code. Your role is to route user requests to the right skill, not to write code.
   - Only use write_file/create_csv_file/create_json_file when the user EXPLICITLY
     asks to save or export specific data. All such files go to the output/ directory.
   - After a spatialclaw call, rely on the tool result for output paths and media metadata.
10. NO SILENT FALLBACK (STRICT): When a user specifies a method and it FAILS:
    - NEVER silently switch to a different method. This is a CRITICAL violation.
    - Report the EXACT error message from the failed method to the user.
    - Ask the user if they want to try an alternative method.
    - Only switch methods with EXPLICIT user confirmation.
    Example: User asks for GraphST deconvolution, it fails → Tell user
    "GraphST deconvolution failed: <error>. Would you like to try Tangram
    or Stereoscope instead?"
    WRONG: Silently run Tangram and say "The Tangram analysis succeeded".
11. MEMORY (IMPORTANT): You have persistent memory across conversations.
    - Use the 'remember' tool to save important context:
      * User preferences: language, default methods, output settings
      * Biological insights: cell type annotations, spatial domains identified
      * Project context: species, tissue type, disease model, research goals
    - Proactively remember when the user:
      * States a preference ("请用中文回答", "use DPI 300")
      * Tells you about their project ("我们研究小鼠大脑的阿尔茨海默病")
      * Confirms a biological annotation ("cluster 0 是T细胞")
    - Your memory context is loaded automatically at the start of each conversation
      under the "## Your Memory" section in the system prompt.
    - Save memory silently.
"""

def build_system_prompt(memory_context: str = "") -> str:
    if SOUL_MD.exists():
        soul = SOUL_MD.read_text(encoding="utf-8")
        logger.info(f"Loaded SOUL.md ({len(soul)} chars)")
    else:
        soul = (
            "You are a spatial transcriptomics AI assistant. "
            "Help users analyse spatial data with clarity and rigour."
        )
        logger.warning("SOUL.md not found, using fallback prompt")

    prompt = f"{soul}\n\n{get_role_guardrails()}"
    if memory_context:
        prompt += f"\n\n## Your Memory\n\n{memory_context}"
    return prompt

SYSTEM_PROMPT: str = ""

def _ensure_system_prompt():
    global SYSTEM_PROMPT
    if not SYSTEM_PROMPT:
        SYSTEM_PROMPT = build_system_prompt()

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

def get_tools() -> list[dict]:
    cli_skill_items = [
        (alias, info)
        for alias, info in registry.skills.items()
        if info.get("interface") != "python-api"
    ]
    skill_names = [alias for alias, _info in cli_skill_items] + ["auto"]
    skill_descriptions = []
    for alias, info in cli_skill_items:
        skill_descriptions.append(f"{alias} ({info.get('description', alias)})")
    skill_desc_text = ", ".join(skill_descriptions)
    skill_arg_help_text = format_skill_extra_arg_help()
    
    return [
        {
            "type": "function",
            "function": {
                "name": "spatialclaw",
                "description": (
                    f"Run a SpatialClaw spatial analysis skill. Available skills: {skill_desc_text}. "
                    "Use mode='demo' only for skills that provide built-in synthetic data. "
                    "Use mode='file' when the user has sent a spatial data file. "
                    "Use mode='path' when user provides a server file/directory path. "
                    "Some skills require directory paths instead of files, and some skills support alternative input flags via extra_args. "
                    "For directory-based input, use mode='path' and set file_path to the sample directory. "
                    "For alternative integration-style inputs, pass flags such as ['--mode', 'integration', '--input-list', '/path/to/samples.txt']. "
                    "IMPORTANT: When this tool returns results, relay the output VERBATIM. "
                    "By default only a text summary is returned (return_media omitted or empty). "
                    "Set return_media ONLY when the user explicitly asks for figures/plots/tables. "
                    "Use 'all' to send everything, or a keyword to filter "
                    "(e.g. 'umap' for UMAP plots, 'qc' for QC violin, 'cluster' for cluster tables). "
                    "Multiple keywords can be comma-separated (e.g. 'umap,qc')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill": {
                            "type": "string",
                            "enum": skill_names,
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["file", "demo", "path"],
                            "description": (
                                "'demo' = built-in synthetic data for supported skills only; "
                                "'file' = user uploaded a file via messaging; "
                                "'path' = user provided a file path on the server."
                            ),
                        },
                        "return_media": {
                            "type": "string",
                            "description": (
                                "Filter for which figures/tables to send back. "
                                "Omit or leave empty for text summary only (default). "
                                "'all' = send all figures and tables. "
                                "Otherwise a comma-separated list of keywords to match filenames "
                                "(e.g. 'umap', 'qc', 'violin', 'cluster', 'umap,qc'). "
                                "Only set when the user explicitly asks for visual results."
                            ),
                        },
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Server-side file path, filename, or directory path for mode='path'. "
                                "For spatial-omics-integrate this is the primary omics file; "
                                "pass --omics2 in extra_args for the secondary modality."
                            ),
                        },
                        "output_dir": {
                            "type": "string",
                            "description": (
                                "Optional absolute or workspace-relative output directory. "
                                "Set this when the user explicitly asks to save results to a specific path."
                            ),
                        },
                        "query": {
                            "type": "string",
                            "description": "Natural language query for auto-routing.",
                        },
                        "extra_args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Additional skill-specific CLI flags as a flat list. "
                                "The tool supplies primary input and output paths from file_path/output_dir; "
                                "extra_args is for registry-whitelisted options on the selected skill. Examples:\n"
                                "- spatial-deconvolution: ['--reference', 'data/scRNA.h5ad', '--cell-type-key', 'CellType']\n"
                                "- spatial-enrichment:   ['--analysis-type', 'go', '--go-ontology', 'BP', '--groupby', 'leiden', '--organism', 'human']\n"
                                "- spatial-enrichment:   ['--analysis-type', 'gsea', '--groupby', 'leiden', '--gene-sets', 'GO_Biological_Process_2023']\n"
                                "- spatial-enrichment:   ['--analysis-type', 'kegg', '--groupby', 'leiden', '--organism', 'human']\n"
                                "- spatial-preprocessing: ['--leiden-resolution', '0.5', '--n-top-hvg', '3000']\n"
                                "Prefer the structured JSON fields only for common cross-skill parameters "
                                "(method, platform, device, n_epochs, output_dir). "
                                "For alternative input modes: "
                                "['--mode', 'integration', '--input-list', '/path/to/samples.txt']. "
                                "For multi-file operations: "
                                "['--method', 'spagcn'] for changing method, "
                                "['--omics2', '/path/to/protein.h5ad', '--omics1-type', 'rna', '--omics2-type', 'protein', '--clustering-method', 'leiden'] "
                                "for spatial-omics-integrate, "
                                "['--method', 'pearlst', '--ground-truth', '/path/to/labels.tsv', '--simclr-features', '/path/to/simCLR_representation_resnet50.csv'] "
                                "for spatial-modality-integrate, "
                                "['--n-domains', '7', '--use-morphological', '--pre-epochs', '20', '--epochs', '50', '--pca-components', '30'] "
                                "for DeepST morphology runs, or ['--no-morphological'] for transcriptome-only DeepST runs. "
                                "['--reference', '/path/to/ref.h5ad'] for reference data (spatial-deconvolution). "
                                "For multi-parameter arguments: ['--param1', 'value1', '--param2', 'value2'].\n"
                                f"{skill_arg_help_text}"
                            ),
                        },
                        "method": {
                            "type": "string",
                            "description": (
                                "Analysis method override passed as --method. "
                                "Examples: spatial-omics-integrate -> spatialglue or spaddm; "
                                "spatial-modality-integrate -> deepst or pearlst."
                            ),
                        },
                        "platform": {
                            "type": "string",
                            "description": (
                                "Optional platform / assay / data-type value for the chosen skill. "
                                "Only set it when the skill actually exposes a platform-style option."
                            ),
                        },
                        "device": {
                            "type": "string",
                            "description": (
                                "Optional compute device for skills that expose --device. "
                                "Use 'gpu' to prefer CUDA with automatic CPU fallback, or pass "
                                "'cpu', 'cuda', 'cuda:0', etc."
                            ),
                        },
                        "n_epochs": {
                            "type": "integer",
                            "description": (
                                "Number of training epochs for deep learning methods. "
                                "Defaults per method if omitted: "
                                "spatial-deconvolution tangram=1000, stereoscope=200, graphst=1000; "
                                "other skills use their script defaults. "
                                "Only set when the user explicitly requests a custom epoch count."
                            ),
                        },
                    },
                    "required": ["skill", "mode"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_file",
                "description": "Save a file that was sent via messaging to a specific folder. Default: SpatialClaw data/ directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination_folder": {"type": "string", "description": "Folder path (absolute)."},
                        "filename": {"type": "string", "description": "Optional filename."},
                    },
                    "required": ["destination_folder"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file with the given content. Files are saved to the output/ directory by default. ONLY use when user explicitly asks to create/save a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Full text content."},
                        "filename": {"type": "string", "description": "Filename with extension."},
                        "destination_folder": {"type": "string", "description": "Folder path (absolute). Default: SpatialClaw data/."},
                    },
                    "required": ["content", "filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_audio",
                "description": "Generate an MP3 audio file from text using edge-tts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to convert to speech."},
                        "filename": {"type": "string", "description": "Output MP3 filename."},
                        "voice": {"type": "string", "description": "TTS voice. Default: en-GB-RyanNeural."},
                        "rate": {"type": "string", "description": "Speech rate. Default: '-5%'."},
                        "destination_folder": {"type": "string", "description": "Output folder."},
                    },
                    "required": ["text", "filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List contents of a directory. Use when user wants to see files in a folder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path (default: current data directory)"}
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_file",
                "description": "Display contents of a CSV, JSON, or TXT file. Use when user wants to view file contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file"},
                        "lines": {"type": "integer", "description": "Number of lines to show (default: 20)"}
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "download_file",
                "description": "Download a file from a URL. Use when user provides a direct file URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "File URL"},
                        "destination": {"type": "string", "description": "Destination path (optional)"}
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_json_file",
                "description": "Create a JSON file from structured data. Saved to output/ by default. ONLY use when user explicitly asks to save data as JSON.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to save as JSON"},
                        "filename": {"type": "string", "description": "Filename (without extension)"},
                        "destination": {"type": "string", "description": "Destination folder (optional)"}
                    },
                    "required": ["data", "filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_csv_file",
                "description": "Create a CSV file from tabular data. Saved to output/ by default. ONLY use when user explicitly asks to save data as CSV.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Array of row objects", "items": {"type": "object", "additionalProperties": True}},
                        "filename": {"type": "string", "description": "Filename (without extension)"},
                        "destination": {"type": "string", "description": "Destination folder (optional)"}
                    },
                    "required": ["data", "filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "make_directory",
                "description": "Create a new directory under output/. ONLY use when user explicitly asks to create a folder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path to create"}
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move_file",
                "description": "Move or rename a file. ONLY use when user explicitly asks to move or rename files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source file path"},
                        "destination": {"type": "string", "description": "Destination path"}
                    },
                    "required": ["source", "destination"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remove_file",
                "description": "Delete a file or directory. ONLY use when user explicitly asks to remove files/folders.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File or directory path to remove"}
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_file_size",
                "description": "Get file size in MB. Use when user asks about file size.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "File path"}
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Save important information to persistent memory so you can recall it "
                    "in future conversations. Use this to remember: user preferences "
                    "(short default settings like language, response style, default methods, DPI), biological insights "
                    "(cell type annotations, spatial domains found), and project context "
                    "(research goals, species, tissue type, disease model). "
                    "Memory persists across conversations and bot restarts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "enum": ["preference", "insight", "project_context"],
                            "description": (
                                "Type of memory to save. "
                                "'preference' = user settings (language, default method, DPI). "
                                "'insight' = biological discovery or durable analysis fact that must be recalled exactly "
                                "(cell types, clusters, output directories, figure filenames). "
                                "'project_context' = research context (species, tissue, disease, goal)."
                            ),
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "For preference: short tendency/default name (e.g. 'language', 'default_method', 'dpi'). "
                                "For insight: entity ID or durable fact key (e.g. 'cluster_0', 'domain_3', 'deepst_result_files'). "
                                "For project_context: not used."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "For preference: short scalar value only (e.g. 'Chinese', 'tangram', '300'), not long factual notes. "
                                "For insight: exact label or factual value. Preserve exact paths and filenames when the user needs later recall. "
                                "For project_context: not used."
                            ),
                        },
                        "domain": {
                            "type": "string",
                            "description": "For preference: scope of the setting (e.g. 'global', 'spatial-preprocessing'). Default: 'global'.",
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "For insight: type of entity or durable fact (e.g. 'cluster', 'spatial_domain', 'cell_type', 'analysis_output').",
                        },
                        "source_analysis_id": {
                            "type": "string",
                            "description": "For insight: ID of the analysis that produced this insight (optional). If omitted, the most recent relevant analysis in memory will be linked when possible.",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["user_confirmed", "ai_predicted"],
                            "description": "For insight: confidence level. Use 'user_confirmed' when user explicitly states a label.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "For insight: optional supporting evidence such as marker genes, manual review, cited observations, or exact output file list.",
                        },
                        "project_goal": {
                            "type": "string",
                            "description": "For project_context: research goal/objective.",
                        },
                        "species": {
                            "type": "string",
                            "description": "For project_context: species (e.g. 'human', 'mouse').",
                        },
                        "tissue_type": {
                            "type": "string",
                            "description": "For project_context: tissue type (e.g. 'brain', 'liver', 'tumor').",
                        },
                        "disease_model": {
                            "type": "string",
                            "description": "For project_context: disease model (e.g. 'breast cancer', 'Alzheimer').",
                        },
                    },
                    "required": ["memory_type"],
                },
            },
        },
    ]

TOOLS = get_tools()

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


def sanitize_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r"[\x00-\x1f]", "", filename)
    filename = filename.replace("..", "").replace("/", "").replace("\\", "")
    return filename or "unnamed_file"


def resolve_dest(folder: str | None, default: Path | None = None) -> Path:
    fallback = default if default is not None else DATA_DIR
    dest = Path(folder) if folder else fallback
    if not dest.is_absolute():
        dest = PROJECT_ROOT / dest
    try:
        dest.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        logger.warning(f"Path escape blocked: {dest}")
        audit("security", severity="HIGH", detail="path_escape_blocked", attempted_path=str(dest))
        dest = fallback
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def validate_path(filepath: Path, allowed_root: Path) -> bool:
    try:
        filepath.resolve().relative_to(allowed_root.resolve())
        return True
    except ValueError:
        return False


def _resolve_output_target(path_arg: str, default_root: Path = DATA_DIR) -> Path | None:
    if not path_arg:
        return None
    raw_path = Path(path_arg).expanduser()
    target = raw_path if raw_path.is_absolute() else default_root / raw_path
    parent = target.parent.resolve()
    filename = sanitize_filename(target.name)
    if not filename:
        return None
    resolved = parent / filename
    _ensure_trusted_dirs()
    allowed_roots = [root.resolve() for root in TRUSTED_DATA_DIRS]
    if raw_path.is_absolute() and not any(
        resolved == root or resolved.is_relative_to(root)
        for root in allowed_roots
    ):
        return None
    if not raw_path.is_absolute():
        default_resolved = default_root.resolve()
        try:
            resolved.relative_to(default_resolved)
        except ValueError:
            return None
    return resolved


def _is_public_download_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, "only http and https URLs are allowed"
    hostname = parsed.hostname
    if not hostname:
        return False, "URL must include a hostname"
    try:
        infos = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False, "hostname could not be resolved"
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False, "hostname resolved to an invalid address"
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, "URL resolves to a non-public address"
    return True, ""


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    filename = unquote(Path(parsed.path).name)
    return sanitize_filename(filename or "downloaded_file")


# ---------------------------------------------------------------------------
# Trusted data directories + file discovery
# ---------------------------------------------------------------------------

def _build_trusted_dirs() -> list[Path]:
    """Build the list of directories where data files may be read from."""
    dirs = [DATA_DIR, EXAMPLES_DIR, OUTPUT_DIR]
    extra = os.environ.get("SPATIALCLAW_DATA_DIRS", os.environ.get("SPATIALCLAW_DATA_DIRS", ""))
    if extra:
        for d in extra.split(","):
            d = d.strip()
            if d:
                p = Path(d)
                if p.is_absolute() and p.is_dir():
                    dirs.append(p)
                else:
                    logger.warning(f"SPATIALCLAW_DATA_DIRS: ignoring '{d}' (not an absolute directory)")
    return dirs


TRUSTED_DATA_DIRS: list[Path] = []


def _ensure_trusted_dirs():
    global TRUSTED_DATA_DIRS
    if not TRUSTED_DATA_DIRS:
        TRUSTED_DATA_DIRS = _build_trusted_dirs()
        logger.info(f"Trusted data dirs: {[str(d) for d in TRUSTED_DATA_DIRS]}")


def validate_input_path(filepath: str) -> Path | None:
    """Validate that a user-supplied file path points to a real file in a trusted directory.

    Returns resolved Path if valid, None otherwise.
    """
    _ensure_trusted_dirs()
    p = Path(filepath).expanduser()
    if not p.is_absolute():
        # 1. Try relative to project root first (most common case)
        candidate = PROJECT_ROOT / p
        if candidate.exists() and candidate.is_file():
            p = candidate
        else:
            # 2. Try each trusted data directory
            for d in TRUSTED_DATA_DIRS:
                candidate = d / p
                if candidate.exists() and candidate.is_file():
                    p = candidate
                    break
            else:
                # 3. Fall back to DATA_DIR
                p = DATA_DIR / p

    resolved = p.resolve()
    if not resolved.exists() or not resolved.is_file():
        return None

    for trusted in TRUSTED_DATA_DIRS:
        try:
            resolved.relative_to(trusted.resolve())
            return resolved
        except ValueError:
            continue

    # Also allow files anywhere under project root
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
        return resolved
    except ValueError:
        pass

    logger.warning(f"Path not in trusted dirs: {resolved}")
    audit("security", severity="MEDIUM", detail="untrusted_path_rejected", path=str(resolved))
    return None


def validate_directory_path(dirpath: str) -> Path | None:
    """Validate that a user-supplied directory path points to a real directory in a trusted location."""
    _ensure_trusted_dirs()
    p = Path(dirpath).expanduser()
    if not p.is_absolute():
        candidate = PROJECT_ROOT / p
        if candidate.exists() and candidate.is_dir():
            p = candidate
        else:
            for d in TRUSTED_DATA_DIRS:
                candidate = d / p
                if candidate.exists() and candidate.is_dir():
                    p = candidate
                    break
            else:
                p = DATA_DIR / p

    resolved = p.resolve()
    if not resolved.exists() or not resolved.is_dir():
        return None

    for trusted in TRUSTED_DATA_DIRS:
        try:
            resolved.relative_to(trusted.resolve())
            return resolved
        except ValueError:
            continue

    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
        return resolved
    except ValueError:
        pass

    logger.warning(f"Directory path not in trusted dirs: {resolved}")
    audit("security", severity="MEDIUM", detail="untrusted_dir_rejected", path=str(resolved))
    return None


def discover_file(filename_or_pattern: str) -> list[Path]:
    """Search trusted data directories for files matching the given name or glob pattern.

    Returns a list of matching paths, sorted by modification time (newest first).
    """
    _ensure_trusted_dirs()

    # Handle absolute paths directly
    if filename_or_pattern.startswith('/'):
        p = Path(filename_or_pattern)
        if p.is_file():
            return [p]
        return []

    matches: list[Path] = []
    for d in TRUSTED_DATA_DIRS:
        if not d.exists():
            continue
        if "*" in filename_or_pattern or "?" in filename_or_pattern:
            matches.extend(f for f in d.rglob(filename_or_pattern) if f.is_file())
        else:
            exact = d / filename_or_pattern
            if exact.is_file():
                matches.append(exact)
            for f in d.rglob(filename_or_pattern):
                if f.is_file() and f not in matches:
                    matches.append(f)
    matches.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Input Validation Functions (Architecture Layer)
# ---------------------------------------------------------------------------

def extract_extra_arg_value(extra_args: list | None, flag: str) -> str | None:
    """Return the value for a CLI flag from a flat extra_args list."""
    if not extra_args:
        return None

    for i, arg in enumerate(extra_args):
        if arg == flag and i + 1 < len(extra_args):
            return extra_args[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _structured_cli_args(skill_key: str, args: dict) -> list[str]:
    """Build CLI args from normalized structured tool-call fields."""
    return build_skill_llm_cli_args(skill_key, args)


def _filter_extra_args_against_structured(
    extra_args: list | None,
    structured_cli_args: list[str],
) -> list[str]:
    """Drop duplicate flags from extra_args when structured fields already supply them."""
    if not extra_args or not isinstance(extra_args, list):
        return []

    structured_flags = {
        token
        for token in structured_cli_args
        if isinstance(token, str) and token.startswith("--")
    }
    filtered: list[str] = []
    skip_next = False

    for arg in extra_args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--output":
            skip_next = True
            continue
        if isinstance(arg, str) and arg.startswith("--output="):
            continue
        if isinstance(arg, str) and arg.startswith("--"):
            eq_pos = arg.find("=")
            if eq_pos > 0:
                normalized_flag = arg[:eq_pos].replace("_", "-")
                arg = normalized_flag + arg[eq_pos:]
            else:
                normalized_flag = arg.replace("_", "-")
                arg = normalized_flag
            if normalized_flag in structured_flags:
                if eq_pos < 0:
                    skip_next = True
                continue
        filtered.append(arg)

    return filtered


def validate_path_by_kind(path_value: str, kind: str = "file") -> Path | None:
    """Validate a path according to its declared kind."""
    if kind == "directory":
        return validate_directory_path(path_value)
    return validate_input_path(path_value)


def path_matches_kind(path: Path, kind: str = "file") -> bool:
    """Check whether a resolved path matches its declared kind."""
    if kind == "directory":
        return path.is_dir()
    return path.is_file()


def get_skill_input_example(skill_info: dict, key: str, default):
    """Return a registry-provided input example when available."""
    examples = skill_info.get("input_examples", {})
    return examples.get(key, default)


def validate_required_extra_inputs(
    primary_input: str | None,
    extra_args: list | None,
    *,
    skill_name: str,
    required_extra_inputs: list[dict],
    primary_kind: str = "file",
    usage_example: str | None = None,
) -> tuple[bool, str]:
    """Validate a primary --input plus registry-defined required extra inputs."""
    primary_placeholder = "<directory>" if primary_kind == "directory" else "<file>"
    usage_lines = [f"  --input {primary_placeholder}"]
    for spec in required_extra_inputs:
        kind = spec.get("kind", "file")
        placeholder = "<directory>" if kind == "directory" else "<file>"
        usage_lines.append(f"  {spec['flag']} {placeholder}")

    if not primary_input:
        return False, (
            f"❌ {skill_name} requires multiple inputs.\n\n"
            "Required usage:\n"
            + "\n".join(usage_lines)
            + (
                f"\n\nExample:\n  {usage_example}"
                if usage_example
                else ""
            )
        )

    primary_path = validate_path_by_kind(primary_input, primary_kind)
    if primary_path is None or not path_matches_kind(primary_path, primary_kind):
        return False, f"❌ Primary input not found: {primary_input}"

    missing_specs = [
        spec for spec in required_extra_inputs
        if not extract_extra_arg_value(extra_args, spec["flag"])
    ]
    if missing_specs:
        missing_lines = "\n".join(
            f"  - {spec['flag']} ({spec.get('label', 'required input')})"
            for spec in missing_specs
        )
        return False, (
            "❌ Missing required extra input(s).\n\n"
            f"{skill_name} requires:\n"
            + "\n".join(usage_lines)
            + f"\n\nMissing:\n{missing_lines}"
            + (
                f"\n\nExample:\n  {usage_example}"
                if usage_example
                else ""
            )
        )

    for spec in required_extra_inputs:
        raw_value = extract_extra_arg_value(extra_args, spec["flag"])
        if raw_value is None:
            continue
        kind = spec.get("kind", "file")
        resolved = validate_path_by_kind(raw_value, kind)
        if resolved is None or not path_matches_kind(resolved, kind):
            label = spec.get("label", spec["flag"])
            return False, f"❌ {label.capitalize()} not found: {raw_value}"

    return True, ""


def validate_directory_or_list_input(
    input_dir: str | None,
    input_list: str | None,
    *,
    skill_name: str,
    example_dir: str,
    list_example: list[str],
    alternative_flag: str = "--input-list",
) -> tuple[bool, str]:
    """Validate a directory-based --input with an optional list-file alternative."""
    if input_dir:
        dir_path = validate_directory_path(input_dir)
        if dir_path is None:
            return False, (
                f"❌ Sample directory not found: {input_dir}\n\n"
                f"Expected a directory such as:\n  {example_dir}"
            )
        return True, ""

    if input_list:
        list_path = validate_input_path(input_list)
        if list_path is None:
            formatted_examples = "\n".join(f"  {line}" for line in list_example)
            return False, (
                f"❌ Input list file not found: {input_list}\n\n"
                f"Expected format (one directory per line):\n{formatted_examples}"
            )
        return True, ""

    return False, (
        f"❌ {skill_name} requires directory-based input.\n\n"
        "Choose ONE of:\n"
        f"  1. --input {example_dir}\n"
        f"  2. {alternative_flag} <file>\n\n"
        "This skill reads a sample directory or an alternative input-list file "
        "with one sample directory per line."
    )


# ---------------------------------------------------------------------------
# execute_spatialclaw
# ---------------------------------------------------------------------------


# Deep learning methods that may take a long time
DEEP_LEARNING_METHODS = {
    "stereoscope", "tangram", "graphst",
    "spagcn", "stagate", "spatialglue", "spaddm",
    "deepst", "pearlst",
    "scvi", "velovi",
    "scanvi", "cellassign",
}


async def execute_spatialclaw(args: dict, session_id: str = None, chat_id: int | str = 0) -> str:
    """Execute an SpatialClaw skill via subprocess (waits until completion)."""
    skill_key = args.get("skill", "auto")
    args = normalize_skill_llm_args(skill_key, args)
    mode = args.get("mode", "demo")
    query = args.get("query", "")
    file_path_arg = args.get("file_path", "")
    skill_info = registry.skills.get(skill_key, {}) if skill_key != "auto" else {}

    # --- Resolve input file for path mode ---
    resolved_path: Path | None = None
    if mode == "path" or file_path_arg:
        mode = "path"
        if file_path_arg:
            resolved_path = validate_input_path(file_path_arg)
            if resolved_path is None:
                found = discover_file(file_path_arg)
                if found:
                    resolved_path = found[0]
                    if len(found) > 1:
                        listing = "\n".join(f"  - {f}" for f in found[:8])
                        return (
                            f"Multiple files match '{file_path_arg}':\n{listing}\n\n"
                            "Please specify the full path."
                        )
                else:
                    if skill_info.get("input_mode") == "directory":
                        potential_dir = validate_directory_path(file_path_arg)
                        if potential_dir is not None:
                            resolved_path = potential_dir
                        else:
                            example_dir = get_skill_input_example(
                                skill_info,
                                "primary",
                                "/path/to/sample-directory",
                            )
                            alt_flags = sorted(skill_info.get("alternative_input_flags", set()))
                            alt_hint = ""
                            if alt_flags:
                                alt_hint = f"- or `{alt_flags[0]}` in extra_args for alternative mode\n\n"
                            # Not found as file or directory
                            return (
                                f"⚠️ **{skill_key} requires directory-based input only.**\n\n"
                                f"ERROR: Could not find '{file_path_arg}' as file or directory.\n\n"
                                f"Please specify:\n"
                                f"- `file_path`: full path to a sample directory such as `{example_dir}`\n"
                                f"{alt_hint}"
                                "This skill reads a sample directory or an alternative input list."
                            )
                    else:
                        # Standard error for non-directory skills
                        dirs_str = ", ".join(str(d) for d in TRUSTED_DATA_DIRS)
                        return (
                            f"File not found: '{file_path_arg}'\n\n"
                            f"Place your data files in one of these directories:\n{dirs_str}\n\n"
                            "Then tell me the filename and I'll find it automatically."
                        )
            logger.info(f"Resolved input path: {resolved_path}")
            audit("file_resolve", file_path=str(resolved_path), original=file_path_arg)

    # --- Auto-routing via spatial orchestrator ---
    if skill_key == "auto":
        orch_script = PROJECT_ROOT / "skills" / "spatial" / "spatial-orchestrator" / "spatial_orchestrator.py"
        if not orch_script.exists():
            return "Error: spatial orchestrator not found."

        orch_input = query
        if resolved_path:
            orch_input = str(resolved_path)
        elif mode == "file":
            for _cid, info in received_files.items():
                orch_input = info["path"]
                break
        if not orch_input:
            return "Error: skill='auto' requires either a file, a file_path, or a query to route."

        try:
            orch_cmd = [PYTHON, str(orch_script)]
            if query:
                orch_cmd.extend(["--query", query])
            else:
                orch_cmd.extend(["--input", orch_input])
            orch_cmd.extend(["--output", str(OUTPUT_DIR / "orchestrator_auto")])

            proc = await asyncio.create_subprocess_exec(
                *orch_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(orch_script.parent),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                return f"Orchestrator error: {stderr.decode()[-500:]}"

            result_json = OUTPUT_DIR / "orchestrator_auto" / "result.json"
            if result_json.exists():
                routing = json.loads(result_json.read_text())
                detected = (
                    routing.get("summary", {}).get("routed_to")
                    or routing.get("data", {}).get("detected_skill", "")
                )
                if detected:
                    skill_key = detected
                    args = normalize_skill_llm_args(skill_key, args)
                    skill_info = registry.skills.get(skill_key, {})
                    logger.info(f"Auto-routed to: {skill_key}")
                else:
                    return f"Orchestrator could not determine a skill. Output: {stdout.decode()[:500]}"
            else:
                return f"Orchestrator completed but no result.json found. stdout: {stdout.decode()[:500]}"
        except asyncio.TimeoutError:
            return "Error: orchestrator timed out."
        except Exception as e:
            return f"Error running orchestrator: {e}"

    # --- Resolve input for file/path mode ---
    input_path = str(resolved_path) if resolved_path else None
    session_path = None

    if not input_path and session_id:
        file_info = received_files.get(session_id)
        if file_info:
            input_path = file_info.get("path")
            session_path = file_info.get("session_path")

    # Get skill metadata from registry
    extra_args = args.get("extra_args", [])
    skill_info = registry.skills.get(skill_key, {})
    requested_output_dir = args.get("output_dir") or ""
    method = str(args.get("method", "")).strip()
    platform_value = str(args.get("platform", "")).strip()
    skill_requires_input = skill_info.get("requires_input", True)
    required_extra_inputs = skill_info.get("required_extra_inputs", [])
    alternative_input_flags = set(skill_info.get("alternative_input_flags", set()))
    structured_cli_args = _structured_cli_args(skill_key, args)
    filtered_extra_args = _filter_extra_args_against_structured(extra_args, structured_cli_args)
    preflight_cli_args = structured_cli_args + filtered_extra_args
    
    # Registry-driven preflight validation for skill input contracts.
    if mode != "demo" and required_extra_inputs:
        is_valid, error_msg = validate_required_extra_inputs(
            input_path,
            preflight_cli_args,
            skill_name=skill_key,
            required_extra_inputs=required_extra_inputs,
            primary_kind=skill_info.get("input_mode", "file"),
            usage_example=get_skill_input_example(skill_info, "usage", None),
        )
        if not is_valid:
            return f"⚠️ **Input Validation Failed**\n\n{error_msg}"
    
    if skill_info.get("input_mode") == "directory":
        example_dir = get_skill_input_example(
            skill_info,
            "primary",
            "/path/to/sample-directory",
        )
        alt_flags = sorted(alternative_input_flags)
        list_flag = alt_flags[0] if alt_flags else "--input-list"
        list_example = get_skill_input_example(
            skill_info,
            list_flag,
            ["path/to/sample1", "path/to/sample2"],
        )
        if mode == "demo":
            if not skill_info.get("demo_args"):
                return (
                    "⚠️ **Input Validation Failed**\n\n"
                    f"{skill_key} does not provide demo data.\n\n"
                    "Use one of:\n"
                    f"  1. --input {example_dir}\n"
                    f"  2. {list_flag} <file>"
                )
        else:
            input_dir = input_path if input_path and Path(input_path).is_dir() else None
            input_list = None
            if list_flag in alternative_input_flags:
                input_list = extract_extra_arg_value(preflight_cli_args, list_flag)

            is_valid, error_msg = validate_directory_or_list_input(
                input_dir=input_dir,
                input_list=input_list,
                skill_name=skill_key,
                example_dir=example_dir,
                list_example=list_example,
                alternative_flag=list_flag,
            )
            if not is_valid:
                return f"⚠️ **Input Validation Failed**\n\n{error_msg}"
    
    # Generic file requirement check for other skills
    if skill_requires_input and mode in ("file", "path") and not input_path and not session_path:
        # Check if skill has alternative inputs
        if required_extra_inputs:
            has_required_extras = all(
                extract_extra_arg_value(preflight_cli_args, spec["flag"])
                for spec in required_extra_inputs
            )
        elif alternative_input_flags:
            has_required_extras = any(
                extract_extra_arg_value(preflight_cli_args, flag) for flag in alternative_input_flags
            )
        else:
            has_required_extras = False
        
        if not has_required_extras:
            _ensure_trusted_dirs()
            dirs_str = ", ".join(str(d) for d in TRUSTED_DATA_DIRS)
            return (
                "No input file available. You can either:\n"
                "1. Upload a file via messaging (if small enough)\n"
                f"2. Place your file in a data directory ({dirs_str}) "
                "and tell me the filename\n"
                "3. Provide the full server path to the file\n"
                "4. Use alternative input parameters if the skill supports them"
            )

    # Output directory
    import uuid
    if requested_output_dir:
        out_dir = Path(requested_output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (Path.cwd() / out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / f"{skill_key}_{ts}_{uuid.uuid4().hex[:8]}"

    # Build command
    cmd = [PYTHON, str(SPATIALCLAW_PY), "run"]
    cmd.append(skill_key)

    if mode == "demo":
        if not skill_info.get("demo_args"):
            return (
                "⚠️ **Input Validation Failed**\n\n"
                f"{skill_key} does not support demo mode."
            )
        cmd.append("--demo")
    elif input_path:
        cmd.extend(["--input", str(input_path)])

    cmd.extend(["--output", str(out_dir)])
    cmd.extend(structured_cli_args)
    cmd.extend(filtered_extra_args)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Log start for deep learning methods
        is_dl = method.lower() in DEEP_LEARNING_METHODS
        if is_dl:
            logger.info(f"Starting {skill_key} with {method} (no timeout, may take 10-60 minutes)")

        # Wait until completion — no timeout
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout_str = stdout_bytes.decode(errors="replace")
        stderr_str = stderr_bytes.decode(errors="replace")
    except Exception as e:
        import traceback as _tb
        # Clean up empty output directory on crash
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        return f"{skill_key} crashed:\n{_tb.format_exc()[-1500:]}"

    if proc.returncode != 0:
        err = stderr_str[-1500:] if stderr_str else stdout_str[-1500:] if stdout_str else "unknown error"
        # Clean up empty output directory on failure
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        # Capture failed analysis to memory (so we remember what was tried)
        if session_id:
            await _auto_capture_analysis(session_id, skill_key, args, None, False)
        return f"{skill_key} failed (exit {proc.returncode}):\n{err}"

    # Collect report + figures from output directory
    return_media = str(args.get("return_media", "")).strip().lower()
    figure_names = []
    table_names = []
    sent_names = []
    if out_dir.exists():
        media_items = []
        for f in sorted(out_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix in (".md", ".html"):
                media_items.append({"type": "document", "path": str(f)})
            elif f.suffix == ".png":
                media_items.append({"type": "photo", "path": str(f)})
                figure_names.append(f.name)
            elif f.suffix == ".csv":
                media_items.append({"type": "document", "path": str(f)})
                table_names.append(f.name)

        if return_media and media_items:
            if return_media == "all":
                filtered = media_items
            else:
                keywords = [k.strip() for k in return_media.split(",") if k.strip()]
                filtered = [
                    item for item in media_items
                    if any(kw in Path(item["path"]).stem.lower() for kw in keywords)
                ]
            if filtered:
                pending_media[session_id] = pending_media.get(session_id, []) + filtered
                sent_names = [Path(item["path"]).name for item in filtered]
                logger.info(f"return_media='{return_media}': sending {len(filtered)}/{len(media_items)} items")

    # Read report for chat display
    report_text = ""
    if out_dir.exists():
        for pattern in ["report.md", "*_report.md", "*.md"]:
            for md_file in sorted(out_dir.glob(pattern)):
                if md_file.name.startswith("."):
                    continue
                report_text = md_file.read_text(encoding="utf-8")
                break
            if report_text:
                break
    # Auto-capture dataset + analysis memory（移到这里，不管有没有 report_text 都执行）
    if session_id:
        if input_path:
            await _auto_capture_dataset(session_id, input_path, platform_value)
        await _auto_capture_analysis(session_id, skill_key, args, out_dir, True)
        
    if not report_text:
        return stdout_str if stdout_str else f"{skill_key} completed. Output: {out_dir}"

    # Trim verbose sections for chat readability; full report is on disk.
    keep_lines = []
    skip = False
    for line in report_text.split("\n"):
        if line.startswith("## Methods") or line.startswith("## Reproducibility"):
            skip = True
        elif line.startswith("## Disclaimer"):
            skip = False
        if line.startswith("!["):
            continue
        if not skip:
            keep_lines.append(line)

    result_text = "\n".join(keep_lines).strip()
    max_report_chars = 3500
    if len(result_text) > max_report_chars:
        compact_lines: list[str] = []
        current_len = 0
        for line in keep_lines:
            line_len = len(line) + 1
            if current_len + line_len > max_report_chars:
                break
            compact_lines.append(line)
            current_len += line_len
            if line.startswith("## Parameters Used"):
                break
        result_text = "\n".join(compact_lines).strip()
        report_path = out_dir / "report.md"
        result_text += (
            "\n\n---\n"
            f"[Report truncated for chat brevity. Full report is saved at: {report_path}]"
        )

    # Append media delivery status so the LLM knows what happened.
    all_names = figure_names + table_names
    if sent_names:
        result_text += (
            "\n\n---\n"
            f"[MEDIA DELIVERY: {len(sent_names)} file(s) already queued for the user: "
            f"{', '.join(sent_names)}. These files will be delivered automatically.]"
        )
        unsent = [n for n in all_names if n not in sent_names]
        if unsent:
            result_text += (
                f"\n[Other available outputs not requested: {', '.join(unsent)}.]"
            )
    elif not return_media and all_names:
        hints = []
        if figure_names:
            hints.append(f"Figures: {', '.join(figure_names)}")
        if table_names:
            hints.append(f"Tables: {', '.join(table_names)}")
        result_text += (
            "\n\n---\n"
            f"[Available outputs: {'; '.join(hints)}. "
            "Tell the user they can request specific figures or tables by name if interested.]"
        )

    return result_text


# ---------------------------------------------------------------------------
# execute_save_file
# ---------------------------------------------------------------------------


async def execute_save_file(args: dict) -> str:
    file_info = None
    for _cid, info in received_files.items():
        file_info = info
        break

    if not file_info:
        return "No recently received file to save. Send a file first."

    src_path = Path(file_info["path"])
    if not src_path.exists():
        return "The temporary file has expired. Please send it again."

    dest_path = resolve_dest(args.get("destination_folder"))
    filename = sanitize_filename(args.get("filename") or file_info["filename"])
    final_path = dest_path / filename

    if not validate_path(final_path, dest_path):
        return f"Error: filename '{filename}' would escape the destination directory."

    shutil.copy2(str(src_path), str(final_path))
    logger.info(f"Saved file: {final_path}")
    try:
        src_path.unlink()
    except OSError:
        pass
    return f"File saved to {final_path}"


# ---------------------------------------------------------------------------
# execute_write_file
# ---------------------------------------------------------------------------


async def execute_write_file(args: dict) -> str:
    content = args.get("content")
    filename = args.get("filename")
    if not content:
        return "Error: 'content' is required."
    if not filename:
        return "Error: 'filename' is required."

    dest = resolve_dest(args.get("destination_folder"), default=OUTPUT_DIR)
    filename = sanitize_filename(filename)
    filepath = dest / filename

    if not validate_path(filepath, dest):
        return f"Error: filename '{filename}' would escape the destination directory."

    filepath.write_text(content, encoding="utf-8")
    logger.info(f"Wrote file: {filepath} ({len(content)} chars)")
    return f"File written to {filepath} ({len(content)} chars)"


# ---------------------------------------------------------------------------
# execute_generate_audio
# ---------------------------------------------------------------------------


async def execute_generate_audio(args: dict) -> str:
    text = args.get("text")
    filename = args.get("filename")
    if not text:
        return "Error: 'text' is required."
    if not filename:
        return "Error: 'filename' is required."
    if not filename.endswith(".mp3"):
        filename += ".mp3"

    filename = sanitize_filename(filename)
    voice = args.get("voice", "en-GB-RyanNeural")
    rate = args.get("rate", "-5%")
    dest = resolve_dest(args.get("destination_folder"))
    filepath = dest / filename

    if not validate_path(filepath, dest):
        return f"Error: filename '{filename}' would escape the destination directory."

    text_path = dest / f".tmp_{filename}.txt"
    text_path.write_text(text, encoding="utf-8")

    try:
        proc = await asyncio.create_subprocess_exec(
            "edge-tts",
            "--voice", voice,
            f"--rate={rate}",
            "--file", str(text_path),
            "--write-media", str(filepath),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        try:
            text_path.unlink()
        except OSError:
            pass

        if proc.returncode != 0:
            err = stderr.decode()[-300:] if stderr else "unknown error"
            return f"Audio generation failed (exit {proc.returncode}): {err}"

        size_mb = filepath.stat().st_size / (1024 * 1024)
        word_count = len(text.split())
        est_minutes = word_count / 150
        logger.info(f"Generated audio: {filepath} ({size_mb:.1f} MB)")
        return f"Audio saved to {filepath} ({size_mb:.1f} MB, ~{word_count} words, ~{est_minutes:.0f} min)"

    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        try:
            text_path.unlink()
        except OSError:
            pass
        return "Audio generation timed out after 5 minutes."
    except FileNotFoundError:
        try:
            text_path.unlink()
        except OSError:
            pass
        return "edge-tts not found. Install with: pip install edge-tts"


# ---------------------------------------------------------------------------
# execute_list_directory
# ---------------------------------------------------------------------------


async def execute_list_directory(args: dict) -> str:
    """List directory contents (restricted to trusted directories)."""
    path_arg = args.get("path", "")
    target_path = Path(path_arg) if path_arg else DATA_DIR

    if not target_path.is_absolute():
        target_path = DATA_DIR / target_path

    # Validate against trusted directories
    _ensure_trusted_dirs()
    resolved = target_path.resolve()
    if not any(
        resolved == td.resolve() or str(resolved).startswith(str(td.resolve()) + os.sep)
        for td in TRUSTED_DATA_DIRS
    ):
        dirs_str = ", ".join(str(d) for d in TRUSTED_DATA_DIRS)
        return f"Access denied: {target_path} is not in trusted directories ({dirs_str})"

    if not target_path.exists():
        return f"Directory not found: {target_path}"

    if not target_path.is_dir():
        return f"Not a directory: {target_path}"

    try:
        items = []
        for item in sorted(target_path.iterdir()):
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                size = item.stat().st_size / (1024 * 1024)
                items.append(f"📄 {item.name} ({size:.2f} MB)")

        if not items:
            return f"Empty directory: {target_path}"

        return f"Contents of {target_path}:\n" + "\n".join(items[:50])
    except Exception as e:
        return f"Error listing directory: {e}"


# ---------------------------------------------------------------------------
# execute_inspect_file
# ---------------------------------------------------------------------------


async def execute_inspect_file(args: dict) -> str:
    """Inspect file contents."""
    file_path_arg = args.get("file_path", "")
    lines_limit = args.get("lines", 20)

    if not file_path_arg:
        return "Error: file_path is required."

    file_path = validate_input_path(file_path_arg)
    if not file_path:
        return f"File not found or not accessible: {file_path_arg}"

    try:
        suffix = file_path.suffix.lower()
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        preview = "\n".join(lines[:lines_limit])
        total = len(lines)

        return f"File: {file_path.name}\nShowing {min(lines_limit, total)} of {total} lines:\n\n{preview}"
    except Exception as e:
        return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# execute_download_file
# ---------------------------------------------------------------------------


async def execute_download_file(args: dict) -> str:
    """Download file from URL."""
    url = args.get("url", "")
    dest_arg = args.get("destination", "")

    if not url:
        return "Error: url is required."

    try:
        allowed, reason = _is_public_download_url(url)
        if not allowed:
            audit("security", severity="HIGH", detail="download_url_blocked", url=url, reason=reason)
            return f"Download blocked: {reason}."

        filename = _filename_from_url(url)
        dest_dir = resolve_dest(dest_arg) if dest_arg else DATA_DIR
        dest_path = dest_dir / filename
        if not validate_path(dest_path, dest_dir):
            return f"Error: filename '{filename}' would escape the destination directory."

        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        length = response.headers.get("Content-Length", "")
        if length:
            try:
                if int(length) > MAX_DOWNLOAD_BYTES:
                    return (
                        "Download blocked: remote file exceeds "
                        f"{MAX_DOWNLOAD_BYTES / (1024 * 1024):.0f} MB limit."
                    )
            except ValueError:
                pass

        bytes_written = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > MAX_DOWNLOAD_BYTES:
                    f.close()
                    dest_path.unlink(missing_ok=True)
                    return (
                        "Download blocked: remote file exceeds "
                        f"{MAX_DOWNLOAD_BYTES / (1024 * 1024):.0f} MB limit."
                    )
                f.write(chunk)

        size_mb = dest_path.stat().st_size / (1024 * 1024)
        return f"Downloaded: {dest_path} ({size_mb:.2f} MB)"
    except Exception as e:
        return f"Download failed: {e}"


# ---------------------------------------------------------------------------
# execute_create_json_file
# ---------------------------------------------------------------------------


async def execute_create_json_file(args: dict) -> str:
    """Create JSON file from data."""
    data = args.get("data", {})
    filename = args.get("filename", "")
    dest_arg = args.get("destination", "")

    if not filename:
        return "Error: filename is required."

    filename = sanitize_filename(filename)
    if not filename.endswith(".json"):
        filename += ".json"

    dest_dir = resolve_dest(dest_arg, default=OUTPUT_DIR) if dest_arg else OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    filepath = dest_dir / filename

    try:
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return f"JSON file created: {filepath}"
    except Exception as e:
        return f"Error creating JSON file: {e}"


# ---------------------------------------------------------------------------
# execute_create_csv_file
# ---------------------------------------------------------------------------


async def execute_create_csv_file(args: dict) -> str:
    """Create CSV file from tabular data."""
    data = args.get("data", [])
    filename = args.get("filename", "")
    dest_arg = args.get("destination", "")

    if not filename:
        return "Error: filename is required."
    if not data:
        return "Error: data is required."

    filename = sanitize_filename(filename)
    if not filename.endswith(".csv"):
        filename += ".csv"

    dest_dir = resolve_dest(dest_arg, default=OUTPUT_DIR) if dest_arg else OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    filepath = dest_dir / filename

    try:
        import csv
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            if isinstance(data[0], dict):
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = csv.writer(f)
                writer.writerows(data)
        return f"CSV file created: {filepath}"
    except Exception as e:
        return f"Error creating CSV file: {e}"


# ---------------------------------------------------------------------------
# execute_make_directory
# ---------------------------------------------------------------------------


async def execute_make_directory(args: dict) -> str:
    """Create a new directory (restricted to trusted directories)."""
    path_arg = args.get("path", "")

    if not path_arg:
        return "Error: path is required."

    target_path = Path(path_arg)
    if not target_path.is_absolute():
        target_path = OUTPUT_DIR / target_path

    # Validate against trusted directories
    _ensure_trusted_dirs()
    resolved = target_path.resolve() if target_path.exists() else target_path.parent.resolve() / target_path.name
    if not any(
        str(resolved).startswith(str(td.resolve()))
        for td in TRUSTED_DATA_DIRS
    ):
        dirs_str = ", ".join(str(d) for d in TRUSTED_DATA_DIRS)
        return f"Access denied: {target_path} is not in trusted directories ({dirs_str})"
        target_path = DATA_DIR / target_path

    try:
        target_path.mkdir(parents=True, exist_ok=True)
        return f"Directory created: {target_path}"
    except Exception as e:
        return f"Error creating directory: {e}"


# ---------------------------------------------------------------------------
# execute_move_file
# ---------------------------------------------------------------------------


async def execute_move_file(args: dict) -> str:
    """Move or rename a file."""
    source_arg = args.get("source", "")
    dest_arg = args.get("destination", "")

    if not source_arg or not dest_arg:
        return "Error: source and destination are required."

    source_path = validate_input_path(source_arg)
    if not source_path:
        return f"Source file not found: {source_arg}"

    dest_path = _resolve_output_target(dest_arg, default_root=DATA_DIR)
    if not dest_path:
        return f"Access denied: destination is outside trusted directories: {dest_arg}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(source_path), str(dest_path))
        return f"Moved: {source_path} -> {dest_path}"
    except Exception as e:
        return f"Error moving file: {e}"


# ---------------------------------------------------------------------------
# execute_remove_file
# ---------------------------------------------------------------------------


async def execute_remove_file(args: dict) -> str:
    """Remove a file or directory."""
    path_arg = args.get("path", "")

    if not path_arg:
        return "Error: path is required."

    target_path = validate_input_path(path_arg)
    if not target_path:
        return f"Path not found: {path_arg}"

    try:
        if target_path.is_dir():
            shutil.rmtree(target_path)
            return f"Removed directory: {target_path}"
        else:
            target_path.unlink()
            return f"Removed file: {target_path}"
    except Exception as e:
        return f"Error removing: {e}"


# ---------------------------------------------------------------------------
# execute_get_file_size
# ---------------------------------------------------------------------------


async def execute_get_file_size(args: dict) -> str:
    """Get file size."""
    file_path_arg = args.get("file_path", "")

    if not file_path_arg:
        return "Error: file_path is required."

    file_path = validate_input_path(file_path_arg)
    if not file_path:
        return f"File not found: {file_path_arg}"

    try:
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return f"File: {file_path.name}\nSize: {size_mb:.2f} MB ({size_bytes:,} bytes)"
    except Exception as e:
        return f"Error getting file size: {e}"


# ---------------------------------------------------------------------------
# execute_remember — LLM tool for saving persistent memories
# ---------------------------------------------------------------------------


async def execute_remember(args: dict, session_id: str = None) -> str:
    """Save information to persistent memory (preferences, insights, project context)."""
    if not memory_store:
        return "Memory system not enabled. Set SPATIALCLAW_MEMORY_BACKEND=sqlite in .env"
    if not session_id:
        return "Memory save requires an active session (user_id + platform)."

    mem_type = args.get("memory_type", "")

    try:
        if mem_type == "preference":
            from spatialclaw.memory.layered_store import PreferenceMemory

            key = args.get("key", "")
            value = args.get("value", "")
            domain = args.get("domain", "global")

            if not key or not value:
                return "Error: preference requires 'key' and 'value'."

            try:
                key, value = validate_preference_payload(key, value)
            except ValueError as e:
                return (
                    "Error: preference memories must be short user tendencies or defaults, "
                    f"not factual notes. {e}"
                )

            pref = PreferenceMemory(
                domain=domain,
                key=key,
                value=value,
                is_strict=False,
            )
            mem_id = await memory_store.save_memory(session_id, pref)
            logger.info(f"Memory saved: preference {key}={value} (domain={domain})")
            return f"✓ Preference saved: {key} = {value} (scope: {domain})"

        elif mem_type == "insight":
            from spatialclaw.memory.layered_store import InsightMemory

            raw_entity_id = args.get("key", "")
            label = args.get("value", "")
            raw_entity_type = args.get("entity_type", "cluster")
            entity_type, entity_id = _canonicalize_insight_identity(raw_entity_type, raw_entity_id)
            source_id = str(args.get("source_analysis_id", "") or "").strip()
            confidence = args.get("confidence", "ai_predicted")
            evidence = str(args.get("evidence", "") or "").strip()

            if not entity_id or not label:
                return "Error: insight requires 'key' (entity ID) and 'value' (label)."

            existing = await _find_existing_insight_memory(session_id, entity_type, entity_id)
            if existing is not None:
                entity_type = getattr(existing, "entity_type", entity_type)
                entity_id = getattr(existing, "entity_id", entity_id)

            if not source_id and existing is not None and getattr(existing, "source_analysis_id", ""):
                source_id = existing.source_analysis_id

            if not source_id:
                recent_analysis = await _get_latest_analysis_memory(session_id)
                if recent_analysis is not None:
                    source_id = recent_analysis.memory_id

            if existing is not None:
                updates = {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "biological_label": label,
                    "confidence": confidence,
                }
                if source_id:
                    updates["source_analysis_id"] = source_id
                if evidence:
                    updates["evidence"] = evidence
                await memory_store.update_memory(existing.memory_id, updates)
                logger.info(f"Memory updated: insight {entity_type} {entity_id} = {label}")
                return f"✓ Insight updated: {entity_type} '{entity_id}' → {label} ({confidence})"

            insight = InsightMemory(
                source_analysis_id=source_id or "",
                entity_type=entity_type,
                entity_id=entity_id,
                biological_label=label,
                evidence=evidence,
                confidence=confidence,
                rule=f"{entity_type} {entity_id}: {label}.",  # ← 用 biological_label 填 rule
                score=3.0 if confidence == "user_confirmed" else 2.0,  # ← 用户确认的给更高初始分
            )
            mem_id = await memory_store.save_memory(session_id, insight)
            logger.info(f"Memory saved: insight {entity_type} {entity_id} = {label}")
            return f"✓ Insight saved: {entity_type} '{entity_id}' → {label} ({confidence})"

        elif mem_type == "project_context":
            from spatialclaw.memory.layered_store import ProjectContextMemory

            ctx = ProjectContextMemory(
                project_goal=args.get("project_goal", ""),
                species=args.get("species"),
                tissue_type=args.get("tissue_type"),
                disease_model=args.get("disease_model"),
            )

            if not any([ctx.project_goal, ctx.species, ctx.tissue_type, ctx.disease_model]):
                return "Error: project_context requires at least one of: project_goal, species, tissue_type, disease_model."

            mem_id = await memory_store.save_memory(session_id, ctx)
            parts = []
            if ctx.project_goal:
                parts.append(f"Goal: {ctx.project_goal}")
            if ctx.species:
                parts.append(f"Species: {ctx.species}")
            if ctx.tissue_type:
                parts.append(f"Tissue: {ctx.tissue_type}")
            if ctx.disease_model:
                parts.append(f"Disease: {ctx.disease_model}")
            logger.info(f"Memory saved: project context ({', '.join(parts)})")
            return f"✓ Project context saved: {' | '.join(parts)}"

        else:
            return f"Error: unknown memory_type '{mem_type}'. Use: preference, insight, project_context."

    except Exception as e:
        logger.error(f"Memory save failed: {e}", exc_info=True)
        return f"Error saving memory: {e}"


# ---------------------------------------------------------------------------
# Tool executor registry
# ---------------------------------------------------------------------------

TOOL_EXECUTORS = {
    "spatialclaw": execute_spatialclaw,
    "save_file": execute_save_file,
    "write_file": execute_write_file,
    "generate_audio": execute_generate_audio,
    "list_directory": execute_list_directory,
    "inspect_file": execute_inspect_file,
    "download_file": execute_download_file,
    "create_json_file": execute_create_json_file,
    "create_csv_file": execute_create_csv_file,
    "make_directory": execute_make_directory,
    "move_file": execute_move_file,
    "remove_file": execute_remove_file,
    "get_file_size": execute_get_file_size,
    "remember": execute_remember,
}

MAX_TOOL_ITERATIONS = int(os.getenv("SPATIALCLAW_MAX_TOOL_ITERATIONS", "20"))  # Increased from 10, configurable


# ---------------------------------------------------------------------------
# LLM tool loop
# ---------------------------------------------------------------------------


async def llm_tool_loop(
    chat_id: int | str,
    user_content: str | list,
    user_id: str = None,
    platform: str = None,
    progress_fn=None,
    progress_update_fn=None,
    on_tool_call=None,
    on_tool_result=None,
) -> str:
    """
    Run the LLM tool-use loop:
    1. Append user message to history
    2. Call LLM with system prompt + history + tools
    3. If tool_calls -> execute -> append results -> call again
    4. Return final text

    progress_fn: async callable(msg) -> handle. Sends a progress message, returns a handle.
    progress_update_fn: async callable(handle, msg). Updates a previously sent progress message.
    on_tool_call: async callable(tool.name, arguments: dict). Called before a tool executes.
    on_tool_result: async callable(tool.name, result: Any). Called after a tool completes.
    """
    # Handle commands before LLM call
    if isinstance(user_content, str) and user_content.strip().startswith("/"):
        cmd = user_content.strip().lower()

        if cmd == "/clear":
            # Only clear conversation history, keep memory intact
            if chat_id in conversations:
                del conversations[chat_id]
            return "✓ Conversation history cleared. (Memory preserved)"

        elif cmd == "/new":
            # Clear conversation history but keep memory
            if chat_id in conversations:
                del conversations[chat_id]
            return "✓ New conversation started. (Memory preserved)"

        elif cmd == "/forget":
            # Clear both conversation and memory for a complete reset
            if chat_id in conversations:
                del conversations[chat_id]

            if session_manager and user_id and platform:
                session_id = f"{platform}:{user_id}:{chat_id}"
                await memory_store.delete_session(session_id)

            return "✓ Memory and conversation cleared. (Fresh start)"

        elif cmd == "/files":
            try:
                items = []
                for item in sorted(DATA_DIR.iterdir()):
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024 * 1024)
                        ext = item.suffix
                        items.append(f"📄 {item.name} ({size_mb:.2f} MB)")
                if not items:
                    return f"📁 Data directory is empty: {DATA_DIR}"
                return f"📁 Data files ({DATA_DIR}):\n" + "\n".join(items[:20])
            except Exception as e:
                return f"Error listing files: {e}"

        elif cmd == "/outputs":
            try:
                items = []
                if OUTPUT_DIR.exists():
                    for item in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                        if item.is_dir():
                            mtime = datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                            items.append(f"📊 {item.name} ({mtime})")
                if not items:
                    return f"📂 No analysis outputs yet: {OUTPUT_DIR}"
                return f"📂 Recent outputs ({OUTPUT_DIR}):\n" + "\n".join(items[:10])
            except Exception as e:
                return f"Error listing outputs: {e}"

        elif cmd == "/skills":
            return format_skills_table(plain=(platform == "feishu"))

        elif cmd == "/recent":
            try:
                items = []
                if OUTPUT_DIR.exists():
                    for item in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                        if item.is_dir():
                            mtime = datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                            report = item / "report.md"
                            summary = "No report"
                            if report.exists():
                                lines = report.read_text(encoding="utf-8").split("\n")
                                summary = next((l.strip("# ") for l in lines if l.startswith("# ")), "Analysis complete")
                            items.append(f"📊 {item.name}\n   {mtime} - {summary}")
                if not items:
                    return "📂 No recent analyses found"
                return "📂 Last 3 Analyses:\n\n" + "\n\n".join(items)
            except Exception as e:
                return f"Error: {e}"

        elif cmd == "/demo":
            return """🎬 Quick Demo Options:

Run any of these for instant results:
• "run spatial-preprocessing demo"
• "run spatial-domain-identification demo"
• "run spatial-de demo"
• "run spatial-enrichment demo"

Or try: "show me a spatial transcriptomics demo" """

        elif cmd == "/examples":
            return """📚 Usage Examples:

**Data Analysis:**
• "Run spatial-preprocessing on brain_visium.h5ad"
• "Analyze data/sample.h5ad with spatial-domain-identification"
• "Run spatial-enrichment on data/sample.h5ad"

**File Operations:**
• "List files in data directory"
• "Show first 20 lines of results.csv"
• "Download https://example.com/data.h5ad"

**Path Mode (for large files):**
• "分析 data/brain_visium.h5ad"
• "对 /mnt/nas/brain_visium.h5ad 做质量控制" """

        elif cmd == "/status":
            uptime = int(time.time() - BOT_START_TIME)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            return f"""🤖 Bot Status:

• Uptime: {hours}h {minutes}m
• LLM Provider: {LLM_PROVIDER_NAME}
• Model: {SPATIALCLAW_MODEL}
• Active Conversations: {len(conversations)}
• Tools Available: {len(TOOL_EXECUTORS)}
• Skills Loaded: {len(registry.skills)}
• Data Directory: {DATA_DIR}
• Output Directory: {OUTPUT_DIR}"""

        elif cmd == "/version":
            return f"""ℹ️ SpatialClaw Version:

• Project: SpatialClaw Spatial Transcriptomics Analysis Platform
• Domain: Spatial Transcriptomics
• Skills: {len(registry.skills)} analysis skills
• Tools: {len(TOOL_EXECUTORS)} bot tools
• Repository: https://github.com/TianGzlab/SpatialClaw

For updates and documentation, visit the GitHub repository."""

        elif cmd == "/help":
            return """# SpatialClaw Bot Commands

**Quick Commands:**
- `/new` - Start new conversation (memory preserved)
- `/clear` - Clear conversation history (memory preserved)
- `/forget` - Clear conversation + memory (complete reset)
- `/help` - Show this help message
- `/files` - List data files
- `/outputs` - Show recent analysis results
- `/skills` - List all available analysis skills
- `/recent` - Show last 3 analyses
- `/demo` - Run a quick demo
- `/examples` - Show usage examples
- `/status` - Bot status and uptime
- `/version` - Show version info

**Memory System:**
- `/clear` and `/new` preserve your analysis history and preferences
- Only `/forget` completely clears all memory
- Bot remembers your datasets, analyses, and preferences across sessions

**File Operations:**
- "List files in data directory"
- "Show contents of file.csv"
- "Download file from URL"

**Data Analysis:**
- "Run spatial-preprocessing on data.h5ad"
- "Analyze data/sample.h5ad with spatial-domain-identification"

For more info: https://github.com/TianGzlab/SpatialClaw"""

    _ensure_system_prompt()
    if llm is None:
        return "Error: LLM client not initialised. Call core.init() first."

    # Load memory context if session manager available
    memory_context = ""
    session_id = None
    user_text = _extract_message_text(user_content)
    if session_manager and user_id and platform:
        # Ensure session exists (create if first time)
        await session_manager.get_or_create(user_id, platform, str(chat_id))
        session_id = f"{platform}:{user_id}:{chat_id}"
        memory_context = await session_manager.load_context(session_id, user_text)
        if user_text:
            targeted_context = await _build_targeted_memory_context(session_id, user_text)
            if targeted_context:
                memory_context = (
                    f"{targeted_context}\n\n{memory_context}"
                    if memory_context
                    else targeted_context
                )

    if remem_operator and memory_store and session_id and user_text:
        if remem_operator.should_invoke(user_text):
            try:
                remem_result = await remem_operator.prepare_context(
                    session_id=session_id,
                    user_query=user_text,
                    memory_store=memory_store,
                )
                remem_summary = str(remem_result.get("summary", "") or "").strip()
                if remem_summary:
                    remem_block = f"## ReMemR1 Reasoning Context\n\n{remem_summary}"
                    memory_context = f"{memory_context}\n\n{remem_block}" if memory_context else remem_block
                    logger.info(
                        "ReMemR1 context prepared: mode=%s candidates=%s callbacks=%s",
                        remem_result.get("mode", "unknown"),
                        remem_result.get("candidate_count", 0),
                        len(remem_result.get("callbacks", [])),
                    )
            except Exception as e:
                logger.warning(
                    f"ReMemR1 operator failed, falling back to default memory prompt path: {e}"
                )

    # Build system prompt with memory context
    system_prompt = build_system_prompt(memory_context) if memory_context else SYSTEM_PROMPT

    history = conversations.setdefault(chat_id, [])
    _conversation_access[chat_id] = time.time()
    _evict_lru_conversations()

    if isinstance(user_content, str):
        history.append({"role": "user", "content": user_content})
    else:
        oai_parts = []
        for block in user_content:
            if block.get("type") == "text":
                oai_parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image":
                src = block.get("source", {})
                data_uri = f"data:{src['media_type']};base64,{src['data']}"
                oai_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                })
        history.append({"role": "user", "content": oai_parts})

    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    # Sanitise: drop orphaned tool messages
    sanitised: list[dict] = []
    for msg in history:
        if msg.get("role") == "tool":
            if sanitised and sanitised[-1].get("role") == "assistant":
                if sanitised[-1].get("tool_calls"):
                    sanitised.append(msg)
                    continue
            logger.warning("Dropped orphaned tool message from history")
            continue
        sanitised.append(msg)
    history[:] = sanitised

    last_message = None
    _notified_methods: set[str] = set()  # Avoid duplicate progress messages
    for _iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = await llm.chat.completions.create(
                model=SPATIALCLAW_MODEL,
                max_tokens=8192,
                messages=[{"role": "system", "content": system_prompt}] + history,
                tools=TOOLS,
            )
        except APIError as e:
            logger.error(f"LLM API error: {e}")
            return f"Sorry, I'm having trouble thinking right now -- API error: {e}"

        normalised = _normalise_chat_response(response)

        # Accumulate token usage statistics
        _accumulate_usage(normalised.get("usage"))

        last_content = str(normalised.get("content", "") or "")
        tool_calls = normalised.get("tool_calls") or []

        assistant_msg: dict = {"role": "assistant", "content": last_content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls
            ]
        history.append(assistant_msg)
        last_message = assistant_msg

        if not tool_calls:
            return last_content or "(no response)"

        for tc in tool_calls:
            func_name = tc["name"]
            executor = TOOL_EXECUTORS.get(func_name)
            if executor:
                try:
                    func_args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    func_args = {}
                logger.info(f"Tool call: {func_name}({json.dumps(func_args)[:200]})")
                audit("tool_call", chat_id=str(chat_id), tool=func_name,
                      args_preview=json.dumps(func_args, default=str)[:300])

                if on_tool_call:
                    if asyncio.iscoroutinefunction(on_tool_call):
                        await on_tool_call(func_name, func_args)
                    else:
                        on_tool_call(func_name, func_args)

                # Send progress message for deep learning methods (once per method)
                _progress_handle = None
                if func_name == "spatialclaw" and progress_fn:
                    dl_method = (func_args.get("method") or "").lower()
                    if dl_method in DEEP_LEARNING_METHODS and dl_method not in _notified_methods:
                        _notified_methods.add(dl_method)
                        method_display = func_args.get("method", dl_method)
                        _progress_handle = await progress_fn(
                            f"⏳ **{method_display}** is a deep learning method and may take "
                            f"10-60 minutes depending on data size. Please be patient...\n\n"
                            f"💡 The analysis is running on the server, you can leave this "
                            f"chat open and come back later."
                        )

                try:
                    # Pass session_id to tools that need it (spatialclaw, remember)
                    if func_name in ("spatialclaw", "remember") and user_id and platform:
                        session_id = f"{platform}:{user_id}:{chat_id}"
                        if func_name == "spatialclaw":
                            result = await executor(func_args, session_id, chat_id=chat_id)
                        else:
                            result = await executor(func_args, session_id)
                    elif func_name == "spatialclaw":
                        result = await executor(func_args, chat_id=chat_id)
                    else:
                        result = await executor(func_args)

                    # Update progress message on success
                    if _progress_handle and progress_update_fn:
                        method_display = (func_args.get("method") or "analysis")
                        await progress_update_fn(
                            _progress_handle,
                            f"✅ **{method_display}** analysis complete!"
                        )
                except Exception as tool_err:
                    logger.error(f"Tool {func_name} raised: {tool_err}", exc_info=True)
                    audit("tool_error", chat_id=str(chat_id), tool=func_name,
                          error=str(tool_err)[:300])
                    result = f"Error executing {func_name}: {type(tool_err).__name__}: {tool_err}"

                    # Update progress message on failure
                    if _progress_handle and progress_update_fn:
                        method_display = (func_args.get("method") or "analysis")
                        await progress_update_fn(
                            _progress_handle,
                            f"❌ **{method_display}** failed: {type(tool_err).__name__}"
                        )

                if on_tool_result:
                    if asyncio.iscoroutinefunction(on_tool_result):
                        await on_tool_result(func_name, result)
                    else:
                        on_tool_result(func_name, result)
            else:
                result = f"Unknown tool: {func_name}"

            history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result),
            })

    return last_message["content"] if last_message and last_message.get("content") else "(max tool iterations reached)"


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def strip_markup(text: str) -> str:
    """Remove markdown/emoji formatting for plain-text messaging.

    Preserves structural elements like list bullets and code content
    while stripping decorative formatting.
    """
    # Strip internal system annotations (not meant for end-users)
    text = re.sub(r"\n*-{3}\n*", "\n", text)  # Strip --- separators
    text = re.sub(
        r"\[(?:MEDIA DELIVERY|Available outputs|Other available outputs)[^\]]*\]\n*",
        "", text,
    )

    # Convert code blocks to indented text (keep content, remove fences)
    text = re.sub(r"```\w*\n?(.*?)```", r"\1", text, flags=re.DOTALL)

    # Inline formatting → plain text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Markdown links → text only
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

    # Heading markers → plain text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Block quotes → plain text (keep content)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    # List bullets: normalise to "- " (keep structure)
    text = re.sub(r"^[\s]*[*]\s+", "- ", text, flags=re.MULTILINE)

    # Strip emojis
    text = re.sub(
        r"[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        r"\U0000200D\U00002B50\U00002B55\U000023CF\U000023E9-\U000023F3"
        r"\U000023F8-\U000023FA\U0000231A\U0000231B\U00003030\U000000A9"
        r"\U000000AE\U00002122\U00002139\U00002194-\U00002199"
        r"\U000021A9-\U000021AA\U0000FE0F]+",
        "",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
