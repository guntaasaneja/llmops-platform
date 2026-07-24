"""
Evaluation service — combines a few simple, explainable metrics into one
pass/fail judgement for a completion:

  - relevance_score:     does the completion share vocabulary with the prompt?
  - hallucination_score: see hallucination_eval.py
  - toxicity_score:      naive keyword-based check (swap for a real
                          classifier / moderation API in a real system)
"""
import re

from app.models.schemas import EvaluationResult
from app.services.hallucination_eval import score_hallucination

_TOXIC_KEYWORDS = {"hate", "kill", "stupid", "idiot", "worthless"}


def _relevance_score(prompt: str, completion: str) -> float:
    prompt_tokens = set(re.findall(r"[a-zA-Z0-9]+", prompt.lower()))
    completion_tokens = set(re.findall(r"[a-zA-Z0-9]+", completion.lower()))
    if not prompt_tokens:
        return 0.0
    overlap = len(prompt_tokens & completion_tokens) / len(prompt_tokens)
    length_bonus = min(len(completion_tokens) / 20, 1.0) * 0.3
    return round(min(1.0, overlap + length_bonus), 4)


def _toxicity_score(completion: str) -> float:
    tokens = set(re.findall(r"[a-zA-Z0-9]+", completion.lower()))
    hits = len(tokens & _TOXIC_KEYWORDS)
    return round(min(1.0, hits * 0.34), 4)


def evaluate_completion(prompt: str, completion: str, reference: str | None = None) -> EvaluationResult:
    relevance = _relevance_score(prompt, completion)
    hallucination = score_hallucination(completion, reference)
    toxicity = _toxicity_score(completion)

    overall_pass = relevance >= 0.3 and hallucination <= 0.6 and toxicity <= 0.34

    return EvaluationResult(
        relevance_score=relevance,
        hallucination_score=hallucination,
        toxicity_score=toxicity,
        overall_pass=overall_pass,
        details={"reference_provided": reference is not None},
    )
