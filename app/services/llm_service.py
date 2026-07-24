"""
LLM serving layer.

Wraps the underlying model provider call with:
  - response caching (Redis)
  - retry/backoff on transient failures (tenacity)
  - latency measurement (Prometheus)
  - token accounting for cost tracking

This is the heart of serving. It contains:

run_inference() — the main function your /inference route calls. It checks the Redis cache first, calls the LLM provider if there's no cache hit, measures latency, and records Prometheus metrics (cache hits/misses, request counts).
_call_provider() — the actual network call to Groq/OpenAI, wrapped with @retry (from tenacity) so transient failures get retried 3 times with exponential backoff instead of failing immediately.
LLMCallResult — a small class that bundles together the completion, token counts, latency, and cost into one object to return.
"""
import time
import uuid

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.redis_client import get_cached_response, set_cached_response
from app.monitoring.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS_TOTAL,
)
from app.services.cost_tracker import estimate_cost

client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


class LLMCallResult:
    def __init__(
        self,
        completion: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cached: bool,
        request_id: str,
    ):
        self.completion = completion
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.latency_ms = latency_ms
        self.cached = cached
        self.request_id = request_id
        self.estimated_cost_usd = estimate_cost(prompt_tokens, completion_tokens)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
async def _call_provider(model: str, prompt: str, temperature: float, max_tokens: int) -> dict:
    """Actual network call to the LLM provider, retried on transient errors."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.model_timeout_seconds,
    )
    choice = response.choices[0].message.content or ""
    usage = response.usage
    return {
        "completion": choice,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }


async def run_inference(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> LLMCallResult:
    """Main entrypoint used by the /inference API route."""
    model = model or settings.default_model
    request_id = str(uuid.uuid4())
    params = {"temperature": temperature, "max_tokens": max_tokens}

    cached = await get_cached_response(model, prompt, params)
    start = time.perf_counter()

    if cached:
        CACHE_HITS_TOTAL.inc()
        result = LLMCallResult(
            completion=cached["completion"],
            model=model,
            prompt_tokens=cached["prompt_tokens"],
            completion_tokens=cached["completion_tokens"],
            latency_ms=(time.perf_counter() - start) * 1000,
            cached=True,
            request_id=request_id,
        )
        INFERENCE_LATENCY.labels(model=model, cached="true").observe(result.latency_ms / 1000)
        INFERENCE_REQUESTS_TOTAL.labels(model=model, status="success").inc()
        return result

    CACHE_MISSES_TOTAL.inc()
    try:
        raw = await _call_provider(model, prompt, temperature, max_tokens)
    except Exception:
        INFERENCE_REQUESTS_TOTAL.labels(model=model, status="error").inc()
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    INFERENCE_LATENCY.labels(model=model, cached="false").observe(latency_ms / 1000)
    INFERENCE_REQUESTS_TOTAL.labels(model=model, status="success").inc()

    await set_cached_response(model, prompt, params, raw)

    return LLMCallResult(
        completion=raw["completion"],
        model=model,
        prompt_tokens=raw["prompt_tokens"],
        completion_tokens=raw["completion_tokens"],
        latency_ms=latency_ms,
        cached=False,
        request_id=request_id,
    )
