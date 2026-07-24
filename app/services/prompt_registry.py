"""
Prompt versioning registry.

Every time a prompt with a given `name` is created/updated, a new row with
an incremented `version` is inserted rather than mutating the existing row.
This gives full history + rollback + the ability to A/B test prompt
versions in the evaluation pipeline.
Contains the functions that manage the Prompt table in Postgres:

create_prompt_version() — inserts a new version, deactivates the old one
get_active_prompt() — fetches whichever version is currently marked active
get_prompt_history() — lists every version ever created
rollback_prompt() — reactivates an older version

This is what powers all three /prompts routes.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import Prompt


async def create_prompt_version(
    db: AsyncSession, name: str, template: str, description: str | None, tags: list[str]
) -> Prompt:
    existing = await db.execute(
        select(Prompt).where(Prompt.name == name).order_by(Prompt.version.desc())
    )
    latest = existing.scalars().first()
    next_version = (latest.version + 1) if latest else 1

    if latest:
        latest.is_active = False

    prompt = Prompt(
        name=name,
        version=next_version,
        template=template,
        description=description,
        tags=tags,
        is_active=True,
    )
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)
    return prompt


async def get_active_prompt(db: AsyncSession, name: str) -> Prompt | None:
    result = await db.execute(
        select(Prompt).where(Prompt.name == name, Prompt.is_active.is_(True))
    )
    return result.scalars().first()


async def get_prompt_history(db: AsyncSession, name: str) -> list[Prompt]:
    result = await db.execute(
        select(Prompt).where(Prompt.name == name).order_by(Prompt.version.desc())
    )
    return list(result.scalars().all())


async def rollback_prompt(db: AsyncSession, name: str, version: int) -> Prompt | None:
    result = await db.execute(
        select(Prompt).where(Prompt.name == name, Prompt.version == version)
    )
    target = result.scalars().first()
    if not target:
        return None

    current = await get_active_prompt(db, name)
    if current:
        current.is_active = False
    target.is_active = True
    await db.commit()
    await db.refresh(target)
    return target
