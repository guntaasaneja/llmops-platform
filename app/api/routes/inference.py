"""
LLM serving endpoint — the main hot path of the platform.

Flow: rate limit -> resolve prompt version -> call LLM (cache-aware) ->
score hallucination -> log to MLflow -> persist run -> track cost -> respond.
POST /inference — the core route. Takes a prompt, does rate-limiting, resolves prompt version (if given), calls the LLM (with caching), scores hallucination, logs to MLflow, records cost, saves to Postgres, returns the completion
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.redis_client import is_rate_limited
from app.mlflow_tracking.experiment_tracker import log_inference_run
from app.models.db_models import InferenceRun
from app.models.schemas import InferenceRequest, InferenceResponse
from app.services.cost_tracker import record_cost_metrics
from app.services.hallucination_eval import score_hallucination
from app.services.llm_service import run_inference
from app.services.prompt_registry import get_active_prompt

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("", response_model=InferenceResponse)
async def infer(payload: InferenceRequest, request: Request, db: AsyncSession = Depends(get_db)):
    client_id = request.client.host if request.client else "anonymous"
    if await is_rate_limited(client_id, limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    prompt_id = None
    final_prompt = payload.prompt
    if payload.prompt_version:
        prompt_row = await get_active_prompt(db, payload.prompt_version)
        if prompt_row:
            prompt_id = prompt_row.id
            final_prompt = prompt_row.template.format(input=payload.prompt)

    result = await run_inference(
        prompt=final_prompt,
        model=payload.model,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    )

    hallucination_score = score_hallucination(result.completion, payload.reference_answer)

    mlflow_run_id = log_inference_run(
        model=result.model,
        prompt=final_prompt,
        completion=result.completion,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        cost_usd=result.estimated_cost_usd,
        hallucination_score=hallucination_score,
    )

    record_cost_metrics(result.model, result.prompt_tokens, result.completion_tokens, result.estimated_cost_usd)

    db.add(
        InferenceRun(
            request_id=result.request_id,
            prompt_id=prompt_id,
            model=result.model,
            prompt_text=final_prompt,
            completion_text=result.completion,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            latency_ms=result.latency_ms,
            cached=result.cached,
            mlflow_run_id=mlflow_run_id,
        )
    )
    await db.commit()

    return InferenceResponse(
        request_id=result.request_id,
        completion=result.completion,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
        latency_ms=result.latency_ms,
        cached=result.cached,
        hallucination_score=hallucination_score,
        mlflow_run_id=mlflow_run_id,
    )
