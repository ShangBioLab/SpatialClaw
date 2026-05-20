import asyncio
import importlib

import pytest


def test_memory_server_allows_local_api_without_token(monkeypatch):
    monkeypatch.delenv("SPATIALCLAW_MEMORY_HOST", raising=False)
    monkeypatch.delenv("SPATIALCLAW_MEMORY_API_TOKEN", raising=False)
    import spatialclaw.memory.server as server

    server = importlib.reload(server)
    if not server._HAS_FASTAPI:
        pytest.skip("FastAPI is not installed")

    from starlette.testclient import TestClient

    with TestClient(server.app) as client:
        response = client.get("/api/browse/domains")

    assert response.status_code == 200
    domains = response.json()
    assert isinstance(domains, list)
    assert [item["domain"] for item in domains] == ["session", "episodic", "semantic"]


def test_memory_server_rejects_non_loopback_api_without_token(monkeypatch):
    monkeypatch.setenv("SPATIALCLAW_MEMORY_HOST", "0.0.0.0")
    monkeypatch.delenv("SPATIALCLAW_MEMORY_API_TOKEN", raising=False)
    import spatialclaw.memory.server as server

    server = importlib.reload(server)
    if not server._HAS_FASTAPI:
        pytest.skip("FastAPI is not installed")

    from starlette.testclient import TestClient

    with TestClient(server.app) as client:
        response = client.get("/api/browse/domains")

    assert response.status_code == 401
    assert response.json()["detail"] == "Memory API token is not configured"


def test_memory_server_allows_health_without_token(monkeypatch):
    monkeypatch.delenv("SPATIALCLAW_MEMORY_API_TOKEN", raising=False)
    import spatialclaw.memory.server as server

    server = importlib.reload(server)
    if not server._HAS_FASTAPI:
        pytest.skip("FastAPI is not installed")

    from starlette.testclient import TestClient

    with TestClient(server.app) as client:
        response = client.get("/health")

    assert response.status_code in {200, 503}


def test_memory_server_default_host_is_loopback(monkeypatch):
    monkeypatch.delenv("SPATIALCLAW_MEMORY_HOST", raising=False)
    monkeypatch.setenv("SPATIALCLAW_MEMORY_API_TOKEN", "token")
    import spatialclaw.memory.server as server
    import uvicorn

    server = importlib.reload(server)
    captured = {}

    def fake_run(app, host, port, reload):
        captured.update({"app": app, "host": host, "port": port, "reload": reload})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    server.main()

    assert captured["host"] == "127.0.0.1"


def test_memory_server_non_loopback_requires_token(monkeypatch):
    monkeypatch.setenv("SPATIALCLAW_MEMORY_HOST", "0.0.0.0")
    monkeypatch.delenv("SPATIALCLAW_MEMORY_API_TOKEN", raising=False)
    import spatialclaw.memory.server as server

    server = importlib.reload(server)

    with pytest.raises(SystemExit):
        server.main()


def test_bot_move_file_rejects_parent_escape(tmp_path, monkeypatch):
    import bot.core as core

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "input.txt"
    source.write_text("x", encoding="utf-8")

    monkeypatch.setattr(core, "DATA_DIR", data_dir)
    monkeypatch.setattr(core, "EXAMPLES_DIR", tmp_path / "examples")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path / "results")
    monkeypatch.setattr(core, "TRUSTED_DATA_DIRS", [data_dir])

    result = asyncio.run(
        core.execute_move_file({"source": str(source), "destination": "../escape.txt"})
    )

    assert "Access denied" in result
    assert source.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_bot_download_blocks_private_hosts(monkeypatch, tmp_path):
    import bot.core as core

    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    monkeypatch.setattr(core, "TRUSTED_DATA_DIRS", [tmp_path])

    called = False

    def fake_get(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("requests.get should not be called for blocked SSRF URL")

    monkeypatch.setattr(core.requests, "get", fake_get)

    result = asyncio.run(core.execute_download_file({"url": "http://127.0.0.1/secret"}))

    assert "Download blocked" in result
    assert called is False


def test_bot_download_enforces_stream_size_limit(monkeypatch, tmp_path):
    import bot.core as core

    class Response:
        headers = {}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"a" * 6

    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    monkeypatch.setattr(core, "TRUSTED_DATA_DIRS", [tmp_path])
    monkeypatch.setattr(core, "MAX_DOWNLOAD_BYTES", 5)
    monkeypatch.setattr(core, "_is_public_download_url", lambda url: (True, ""))
    monkeypatch.setattr(core.requests, "get", lambda *args, **kwargs: Response())

    result = asyncio.run(core.execute_download_file({"url": "https://example.com/data.txt"}))

    assert "exceeds" in result
    assert not (tmp_path / "data.txt").exists()


def test_notebook_create_rejects_absolute_and_parent_escape(tmp_path):
    from spatialclaw.agents import tools

    tools.set_workspace_dir(str(tmp_path / "workspace"))

    absolute = tools.notebook_create.invoke({"notebook_path": str(tmp_path / "escape.ipynb")})
    parent = tools.notebook_create.invoke({"notebook_path": "../escape.ipynb"})

    assert "must be relative" in absolute
    assert "cannot escape" in parent
    assert not (tmp_path / "escape.ipynb").exists()
