"""
Async SQLAlchemy engine + session factory for PostgreSQL.
Sets up the async SQLAlchemy engine and session that talks to PostgreSQL. It defines:

engine — the actual connection pool to Postgres
AsyncSessionLocal — a factory that creates new DB sessions
get_db() — a FastAPI dependency that route handlers use (db: AsyncSession = Depends(get_db)) to get a session for that one request, and automatically closes it afterward
init_db() — creates all the tables (Prompt, InferenceRun, EvaluationRecord) on startup
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a scoped async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Create tables on startup (in real prod, use Alembic migrations instead)."""
    from app.models import db_models  # noqa: F401  ensure models are registered

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
