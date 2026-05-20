from __future__ import annotations

import asyncio
import sys
import types

from spatialclaw.interactive import interactive


def test_stream_llm_response_uses_visible_session_identity(monkeypatch):
    captured = {}
    fake_core = types.ModuleType("bot.core")
    fake_core.conversations = {}
    fake_core._conversation_access = {}

    def get_usage_snapshot():
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }

    async def llm_tool_loop(chat_id, user_content, **kwargs):
        captured["chat_id"] = chat_id
        captured["user_content"] = user_content
        captured["kwargs"] = kwargs
        fake_core.conversations[chat_id].append(
            {"role": "user", "content": user_content}
        )
        fake_core.conversations[chat_id].append(
            {"role": "assistant", "content": "ok"}
        )
        return "ok"

    fake_core.get_usage_snapshot = get_usage_snapshot
    fake_core.llm_tool_loop = llm_tool_loop
    fake_bot = types.ModuleType("bot")
    fake_bot.core = fake_core
    monkeypatch.setitem(sys.modules, "bot", fake_bot)
    monkeypatch.setitem(sys.modules, "bot.core", fake_core)

    messages = [
        {"role": "assistant", "content": "seed"},
        {"role": "user", "content": "hello"},
    ]

    result = asyncio.run(
        interactive._stream_llm_response(messages, chat_id="session-20260518")
    )

    assert result == "ok"
    assert captured["chat_id"] == "session-20260518"
    assert captured["user_content"] == "hello"
    assert captured["kwargs"]["user_id"] == "local_cli_user"
    assert captured["kwargs"]["platform"] == "cli"
    assert captured["chat_id"] != "__interactive__"
    assert messages[-1] == {"role": "assistant", "content": "ok"}


def test_tui_llm_response_uses_visible_session_identity(monkeypatch, tmp_path):
    from spatialclaw.interactive import tui

    if not tui._HAS_TEXTUAL:
        import pytest

        pytest.skip("textual is not installed")

    captured = {}
    fake_core = types.ModuleType("bot.core")
    fake_core.conversations = {}
    fake_core._conversation_access = {}

    def get_usage_snapshot():
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "input_price_per_1m": 0,
            "output_price_per_1m": 0,
            "model": "fake",
        }

    async def llm_tool_loop(chat_id, user_content, **kwargs):
        captured["chat_id"] = chat_id
        captured["user_content"] = user_content
        captured["kwargs"] = kwargs
        fake_core.conversations[chat_id].append(
            {"role": "user", "content": user_content}
        )
        fake_core.conversations[chat_id].append(
            {"role": "assistant", "content": "ok"}
        )
        return "ok"

    fake_core.get_usage_snapshot = get_usage_snapshot
    fake_core.llm_tool_loop = llm_tool_loop
    fake_bot = types.ModuleType("bot")
    fake_bot.core = fake_core
    monkeypatch.setitem(sys.modules, "bot", fake_bot)
    monkeypatch.setitem(sys.modules, "bot.core", fake_core)

    app = tui.SpatialClawTUI(
        session_id="tui-session-20260518",
        workspace_dir=str(tmp_path),
    )
    app._messages = [
        {"role": "assistant", "content": "seed"},
        {"role": "user", "content": "hello tui"},
    ]
    monkeypatch.setattr(app, "_add_system_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "_add_assistant_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "_update_usage_bar", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tui, "save_session", lambda *_args, **_kwargs: _noop_async())

    asyncio.run(app._llm_response_async())

    assert captured["chat_id"] == "tui-session-20260518"
    assert captured["user_content"] == "hello tui"
    assert captured["kwargs"]["user_id"] == "local_tui_user"
    assert captured["kwargs"]["platform"] == "tui"
    assert captured["chat_id"] != "__tui__"
    assert app._messages[-1] == {"role": "assistant", "content": "ok"}


async def _noop_async():
    return None


def test_single_shot_uses_requested_session_identity(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(interactive, "_init_llm", lambda _config: ("fake-model", "fake-provider"))

    async def fake_stream(messages, chat_id):
        captured["chat_id"] = chat_id
        captured["messages"] = list(messages)
        messages.append({"role": "assistant", "content": "done"})
        return "done"

    async def fake_save_session(session_id, messages, **kwargs):
        captured["saved_session_id"] = session_id
        captured["saved_messages"] = list(messages)
        captured["save_kwargs"] = kwargs

    monkeypatch.setattr(interactive, "_stream_llm_response", fake_stream)
    monkeypatch.setattr(interactive, "save_session", fake_save_session)

    asyncio.run(
        interactive._single_shot(
            prompt="hello",
            workspace_dir=str(tmp_path),
            session_id="memory-session-20260518",
            config={},
        )
    )

    assert captured["chat_id"] == "memory-session-20260518"
    assert captured["saved_session_id"] == "memory-session-20260518"
    assert captured["messages"][0] == {"role": "user", "content": "hello"}
    assert captured["saved_messages"][-1] == {"role": "assistant", "content": "done"}
