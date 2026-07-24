"""
Prompt versioning API — create new prompt versions, list history, roll back.
POST /prompts — creates a new versioned prompt
GET /prompts/{name}/history — lists every version of a named prompt
POST /prompts/{name}/rollback/{version} — reactivates an older version
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.models.schemas import PromptCreateRequest, PromptResponse
from app.services.prompt_registry import (
    create_prompt_version,
    get_prompt_history,
    rollback_prompt,
)

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.post("", response_model=PromptResponse)
async def create_prompt(payload: PromptCreateRequest, db: AsyncSession = Depends(get_db)):
    prompt = await create_prompt_version(
        db, payload.name, payload.template, payload.description, payload.tags
    )
    return prompt


@router.get("/{name}/history", response_model=list[PromptResponse])
async def prompt_history(name: str, db: AsyncSession = Depends(get_db)):
    history = await get_prompt_history(db, name)
    if not history:
        raise HTTPException(status_code=404, detail=f"No prompt found with name '{name}'")
    return history


@router.post("/{name}/rollback/{version}", response_model=PromptResponse)
async def rollback(name: str, version: int, db: AsyncSession = Depends(get_db)):
    prompt = await rollback_prompt(db, name, version)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Version {version} not found for '{name}'")
    return prompt
