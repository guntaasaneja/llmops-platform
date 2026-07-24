"""
One-off script to initialize database tables (for local/dev use).
In production, prefer Alembic migrations for schema changes.

Usage: python scripts/init_db.py
"""
import asyncio

from app.core.db import init_db


async def main():
    print("Creating database tables...")
    await init_db()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
