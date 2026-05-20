# pyright: reportArgumentType=false, reportCallIssue=false

"""
Database connection and session management for SpatialClaw Graph Memory.

Supports both SQLite (local, default) and PostgreSQL (remote).
"""

import os
import shutil
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

from sqlalchemy import text
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# async_sessionmaker added in SQLAlchemy 2.0; fall back to sessionmaker for 1.4+
try:
    from sqlalchemy.ext.asyncio import async_sessionmaker as _async_sessionmaker
except ImportError:
    _async_sessionmaker = None  # type: ignore

from .models import Base


# Default DB URL — uses <project-root>/.config/spatialclaw/memory.db.
_DEFAULT_DB_DIR_PARTS = (".config", "spatialclaw")
_DEFAULT_DB_FILENAME = "memory.db"
_LEGACY_DOMAIN_NAMES = {"episode", "short_term", "long_term"}
try:
    from spatialclaw.paths import PROJECT_ROOT as _PROJECT_ROOT
except Exception:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_default_db_url() -> str:
    """Return the default SQLite database URL under the SpatialClaw project root."""
    base_dir = _PROJECT_ROOT.resolve().joinpath(*_DEFAULT_DB_DIR_PARTS)
    return f"sqlite+aiosqlite:///{base_dir / _DEFAULT_DB_FILENAME}"


def _get_database_url() -> str:
    """Resolve database URL from environment or default."""
    return os.getenv("SPATIALCLAW_MEMORY_DB_URL", _get_default_db_url())


def _sqlite_path_from_url(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite"):
        return None
    if database_url.startswith("sqlite+aiosqlite:///"):
        raw_path = database_url.split("sqlite+aiosqlite:///", 1)[1]
    elif database_url.startswith("sqlite:///"):
        raw_path = database_url.split("sqlite:///", 1)[1]
    else:
        return None
    return Path(raw_path).expanduser().resolve()


def _is_default_database_url(database_url: str) -> bool:
    default_path = _sqlite_path_from_url(_get_default_db_url())
    current_path = _sqlite_path_from_url(database_url)
    return default_path is not None and current_path == default_path


def _inspect_legacy_sqlite_database(db_path: Path) -> list[str]:
    """Return reasons why an existing SQLite DB is not canonical."""
    if not db_path.exists() or db_path.stat().st_size == 0:
        return []

    reasons: list[str] = []
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.DatabaseError as exc:
        return [f"SQLite database cannot be inspected: {exc}"]

    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        if cursor.fetchone() is None:
            return []

        cursor.execute("PRAGMA table_info(memories)")
        memory_columns = {row[1] for row in cursor.fetchall()}
        old_columns = {"is_retired", "superseded_by_id"} & memory_columns
        if old_columns:
            reasons.append(
                "legacy memories columns present: " + ", ".join(sorted(old_columns))
            )

        required_columns = {"deprecated", "migrated_to"}
        missing_columns = required_columns - memory_columns
        if missing_columns:
            reasons.append(
                "canonical memories columns missing: " + ", ".join(sorted(missing_columns))
            )

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='paths'"
        )
        if cursor.fetchone() is not None:
            placeholders = ",".join("?" for _ in _LEGACY_DOMAIN_NAMES)
            cursor.execute(
                f"SELECT DISTINCT domain FROM paths WHERE domain IN ({placeholders})",
                tuple(sorted(_LEGACY_DOMAIN_NAMES)),
            )
            legacy_domains = sorted(row[0] for row in cursor.fetchall())
            if legacy_domains:
                reasons.append(
                    "legacy path domains present: " + ", ".join(legacy_domains)
                )
    finally:
        connection.close()

    return reasons


def _archive_legacy_sqlite_database(db_path: Path, reasons: list[str]) -> Path:
    """Move a legacy default SQLite DB aside so a canonical DB can be created."""
    backup_dir = _PROJECT_ROOT.resolve() / ".run" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"memory-legacy-{timestamp}.db"

    shutil.move(str(db_path), str(backup_path))
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            shutil.move(str(sidecar), str(backup_path.with_name(backup_path.name + suffix)))

    reason_text = "\n".join(f"- {reason}" for reason in reasons)
    backup_path.with_suffix(backup_path.suffix + ".txt").write_text(
        "SpatialClaw archived this legacy memory database because runtime "
        "compatibility has been removed.\n\nReasons:\n"
        f"{reason_text}\n",
        encoding="utf-8",
    )
    return backup_path


class DatabaseManager:
    """Async database connection manager.

    Provides session lifecycle management (commit/rollback) and table creation.
    All business-logic services receive a ``DatabaseManager`` via injection.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or _get_database_url()
        self.db_type = self._detect_database_type(self.database_url)

        # Ensure SQLite directory exists
        if self.db_type == "sqlite":
            db_path = self.database_url.split("///", 1)[-1] if "///" in self.database_url else ""
            if db_path:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        engine_kwargs = {"echo": False}
        if self.db_type == "postgresql":
            parsed = urlparse(self.database_url)
            is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")

            connect_args = {}
            parsed_qs = parse_qs(parsed.query, keep_blank_values=True)
            ssl_values = parsed_qs.get("ssl", []) + parsed_qs.get("sslmode", [])
            ssl_value = ssl_values[-1].lower() if ssl_values else ""
            ssl_disabled = ssl_value in ("disable", "false", "off", "0", "no")

            if not is_local and not ssl_disabled:
                connect_args["ssl"] = "require"
                connect_args["statement_cache_size"] = 0

            engine_kwargs.update(
                {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "pool_recycle": 3600,
                    "pool_pre_ping": True,
                    "connect_args": connect_args,
                }
            )

        self.engine = create_async_engine(self.database_url, **engine_kwargs)

        if self.db_type == "sqlite":
            @event.listens_for(self.engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        if _async_sessionmaker is not None:
            self.async_session = _async_sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
        else:
            # SQLAlchemy 1.4 fallback
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False  # type: ignore
            )

    @staticmethod
    def _detect_database_type(url: str) -> str:
        if "postgresql" in url:
            return "postgresql"
        return "sqlite"

    @asynccontextmanager
    async def session(self):
        """Get an async session context manager."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @asynccontextmanager
    async def _optional_session(self, session: Optional[AsyncSession] = None):
        """Helper to use an existing session or create a new one."""
        if session:
            yield session
        else:
            async with self.session() as new_session:
                yield new_session

    async def init_db(self):
        """Create tables if they don't exist, and ensure root node is present."""
        try:
            from sqlalchemy import inspect as sa_inspect

            if self.db_type == "sqlite":
                db_path = _sqlite_path_from_url(self.database_url)
                if db_path is not None:
                    legacy_reasons = _inspect_legacy_sqlite_database(db_path)
                    if legacy_reasons:
                        if _is_default_database_url(self.database_url):
                            backup_path = _archive_legacy_sqlite_database(
                                db_path, legacy_reasons
                            )
                            await self.engine.dispose()
                            self.engine = create_async_engine(
                                self.database_url, echo=False
                            )
                            @event.listens_for(self.engine.sync_engine, "connect")
                            def set_sqlite_pragma_after_archive(dbapi_connection, connection_record):
                                cursor = dbapi_connection.cursor()
                                cursor.execute("PRAGMA foreign_keys=ON")
                                cursor.close()
                            if _async_sessionmaker is not None:
                                self.async_session = _async_sessionmaker(
                                    self.engine,
                                    class_=AsyncSession,
                                    expire_on_commit=False,
                                )
                            else:
                                self.async_session = sessionmaker(
                                    self.engine,
                                    class_=AsyncSession,
                                    expire_on_commit=False,  # type: ignore
                                )
                            print(
                                "SpatialClaw archived legacy memory database at "
                                f"{backup_path}"
                            )
                        else:
                            joined = "; ".join(legacy_reasons)
                            raise RuntimeError(
                                "Configured memory database is not canonical. "
                                f"{joined}. Archive or migrate it manually before startup."
                            )

            def check_initialized(connection):
                return sa_inspect(connection).has_table("memories")

            async with self.engine.begin() as conn:
                is_initialized = await conn.run_sync(check_initialized)
                if not is_initialized:
                    await conn.run_sync(Base.metadata.create_all)

            # Ensure the root node exists (all edges reference it as parent)
            await self._ensure_root_node()

            # Create FTS5 virtual table for SQLite if not exists
            if self.db_type == "sqlite":
                await self._create_fts_table()

        except Exception as e:
            db_url = self.database_url
            if "@" in db_url and ":" in db_url:
                try:
                    parsed = urlparse(db_url)
                    if parsed.password:
                        db_url = db_url.replace(f":{parsed.password}@", ":***@")
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to connect to database.\n"
                f"  URL: {db_url}\n"
                f"  Error: {e}\n\n"
                f"Troubleshooting:\n"
                f"  - Check SPATIALCLAW_MEMORY_DB_URL in .env\n"
                f"  - For SQLite, ensure the directory exists\n"
                f"  - For PostgreSQL, ensure the host is reachable"
            ) from e

    async def _ensure_root_node(self):
        """Insert the root node into the nodes table if it doesn't exist.

        The graph is a tree rooted at ROOT_NODE_UUID. All top-level edges
        reference it as parent_uuid, so the row in `nodes` MUST exist before
        any edge can be created (due to FOREIGN KEY constraints).
        """
        from sqlalchemy import select
        from .models import Node, ROOT_NODE_UUID

        async with self.async_session() as session:
            result = await session.execute(
                select(Node).where(Node.uuid == ROOT_NODE_UUID)
            )
            if result.scalars().first() is None:
                session.add(Node(uuid=ROOT_NODE_UUID))
                await session.commit()

    async def _create_fts_table(self):
        """Create SQLite FTS5 virtual table for full-text search."""
        from sqlalchemy import text

        async with self.async_session() as session:
            try:
                # Check if FTS table already exists
                result = await session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='search_documents_fts'")
                )
                if result.scalar() is None:
                    await session.execute(
                        text("""
                            CREATE VIRTUAL TABLE IF NOT EXISTS search_documents_fts
                            USING fts5(
                                domain,
                                path,
                                node_uuid,
                                uri,
                                content,
                                disclosure,
                                search_terms,
                                content=search_documents,
                                content_rowid=rowid
                            )
                        """)
                    )
                    await session.commit()
            except Exception:
                # FTS5 may not be available on all SQLite builds
                await session.rollback()

    async def close(self):
        """Close the database connection."""
        await self.engine.dispose()
