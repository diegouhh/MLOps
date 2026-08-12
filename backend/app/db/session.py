from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache
def get_engine():
    settings = get_settings()
    kwargs = (
        {"connect_args": {"check_same_thread": False, "timeout": 30}}
        if settings.database_url.startswith("sqlite")
        else {}
    )
    engine = create_engine(settings.database_url, pool_pre_ping=True, **kwargs)
    if settings.database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(dbapi_connection, connection_record):  # noqa: ARG001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return engine


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_database_caches() -> None:
    get_session_factory.cache_clear()
    get_engine.cache_clear()
