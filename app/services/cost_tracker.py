"""
Cost tracking — converts token usage into estimated USD cost and records it
to Prometheus counters (for dashboards / alerting) and Postgres (for
historical reporting / chargeback per team or API key).
Takes token counts and configurable per-1K-token rates (from .env), does simple math, returns a dollar estimate. Also has record_cost_metrics() which pushes that number into Prometheus counters. This is deliberately kept separate from llm_service.py so the "how much did this cost" logic isn't tangled up with "how do I call the LLM."
"""
from app.core.config import settings
from app.monitoring.metrics import INFERENCE_COST_USD_TOTAL, TOKENS_USED_TOTAL


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    prompt_cost = (prompt_tokens / 1000) * settings.cost_per_1k_prompt_tokens
    completion_cost = (completion_tokens / 1000) * settings.cost_per_1k_completion_tokens
    return round(prompt_cost + completion_cost, 8)


def record_cost_metrics(model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
    TOKENS_USED_TOTAL.labels(model=model, token_type="prompt").inc(prompt_tokens)
    TOKENS_USED_TOTAL.labels(model=model, token_type="completion").inc(completion_tokens)
    INFERENCE_COST_USD_TOTAL.labels(model=model).inc(cost_usd)
