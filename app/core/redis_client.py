"""
Redis client wrapper used for:
  1. Response caching (idempotent prompt+model -> completion)
  2. Simple sliding-window rate limiting per API key / client

Sets up the Redis connection pool and contains two pieces of actual logic:

Caching: cache_key(), get_cached_response(), set_cached_response() — hash the prompt+model+params into a key, check/store completions in Redis
Rate limiting: is_rate_limited() — the fixed-window rate limiter that counts requests per client per 60-second window
"""
import hashlib
import json
import time

import redis.asyncio as redis

from app.core.config import settings

_redis_pool = redis.ConnectionPool(
    host=settings.redis_host, port=settings.redis_port, decode_responses=True
)


def get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=_redis_pool)


def cache_key(model: str, prompt: str, params: dict) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "params": params}, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"llm:cache:{digest}"


async def get_cached_response(model: str, prompt: str, params: dict) -> dict | None:
    r = get_redis()
    key = cache_key(model, prompt, params)
    cached = await r.get(key)
    return json.loads(cached) if cached else None


async def set_cached_response(model: str, prompt: str, params: dict, response: dict) -> None:
    r = get_redis()
    key = cache_key(model, prompt, params)
    await r.set(key, json.dumps(response), ex=settings.redis_ttl_seconds)


async def is_rate_limited(client_id: str, limit: int = 60, window_seconds: int = 60) -> bool:
    """Fixed-window rate limiter: `limit` requests per `window_seconds` per client_id."""
    r = get_redis()
    window = int(time.time() // window_seconds)
    key = f"llm:ratelimit:{client_id}:{window}"
    count = await r.incr(key)
    if count == 1:
        await r.expire(key, window_seconds)
    return count > limit
