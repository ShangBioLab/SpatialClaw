"""
SpatialClaw Memory API Server.

FastAPI application serving the graph memory REST API.
Start with: python -m spatialclaw.memory.server

Requires: pip install fastapi uvicorn
"""

import os
import secrets
from contextlib import asynccontextmanager
from ipaddress import ip_address

# Guard: FastAPI is optional
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


def _build_app():
    """Build and configure the FastAPI application."""
    if not _HAS_FASTAPI:
        return None

    @asynccontextmanager
    async def lifespan(app):
        from . import get_db_manager
        db = get_db_manager()
        await db.init_db()
        yield
        from . import close_db
        await close_db()

    app = FastAPI(
        title="SpatialClaw Memory API",
        description="Graph-based memory system for SpatialClaw",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Bearer token auth. Health stays unauthenticated for readiness checks.
    # Loopback-only local use may run without a token; non-loopback hosts still
    # require SPATIALCLAW_MEMORY_API_TOKEN and are rejected in main().
    _api_token = os.getenv("SPATIALCLAW_MEMORY_API_TOKEN", "")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        if not _api_token:
            if _local_no_token_allowed():
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"detail": "Memory API token is not configured"},
            )
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        provided = auth_header.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(provided, _api_token):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

    # Register API routers
    from .api.browse import router as browse_router
    from .api.review import router as review_router
    from .api.maintenance import router as maintenance_router

    app.include_router(browse_router)
    app.include_router(review_router)
    app.include_router(maintenance_router)

    @app.get("/health", tags=["health"])
    async def health_check():
        from sqlalchemy import text
        from . import get_db_manager
        db_status = "disconnected"
        try:
            db = get_db_manager()
            async with db.session() as session:
                await session.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception:
            pass
        status_code = 200 if db_status == "connected" else 503
        return JSONResponse(
            content={
                "status": "ok" if db_status == "connected" else "degraded",
                "database": db_status,
                "service": "spatialclaw-memory",
            },
            status_code=status_code,
        )

    return app


# Build app at module level (None if fastapi not installed)
app = _build_app()


def _is_loopback_host(host: str) -> bool:
    """Return True when uvicorn will only bind to loopback interfaces."""
    normalized = (host or "").strip().lower()
    if normalized in {"localhost"}:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _local_no_token_allowed() -> bool:
    """Return True when the memory server is configured for local-only use."""
    host = os.getenv("SPATIALCLAW_MEMORY_HOST", "127.0.0.1")
    return _is_loopback_host(host)


def main():
    """Entry point for running the memory API server."""
    if not _HAS_FASTAPI:
        print("ERROR: FastAPI is not installed.")
        print("Install with: pip install fastapi uvicorn")
        raise SystemExit(1)

    import uvicorn

    host = os.getenv("SPATIALCLAW_MEMORY_HOST", "127.0.0.1")
    port = int(os.getenv("SPATIALCLAW_MEMORY_PORT", "8766"))
    api_token = os.getenv("SPATIALCLAW_MEMORY_API_TOKEN", "")

    if not _is_loopback_host(host) and not api_token:
        print(
            "ERROR: SPATIALCLAW_MEMORY_API_TOKEN is required when binding "
            f"memory server to non-loopback host {host!r}."
        )
        raise SystemExit(1)

    print(f"SpatialClaw Memory API starting on http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")

    uvicorn.run(
        "spatialclaw.memory.server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
