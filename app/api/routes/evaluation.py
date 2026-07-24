"""
Evaluation API — score a single prompt/completion pair against a few
simple, explainable metrics (relevance, hallucination, toxicity).
POST /evaluation/score — scores any prompt/completion pair directly (relevance, hallucination, toxicity) without needing to run a full inference call first — useful for testing your evaluation logic or scoring completions from elsewhere
"""
from fastapi import APIRouter

from app.models.schemas import EvaluationRequest, EvaluationResult
from app.services.evaluation_service import evaluate_completion

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/score", response_model=EvaluationResult)
async def score(payload: EvaluationRequest):
    return evaluate_completion(payload.prompt, payload.completion, payload.reference_answer)
