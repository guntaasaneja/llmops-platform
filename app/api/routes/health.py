"""
Health / readiness endpoint — checked by Kubernetes liveness & readiness
probes (see k8s/deployment.yaml) and by CI smoke tests post-deploy.
GET /health — checks Postgres + Redis connectivity, returns overall status (healthy/degraded)
GET /ready — simple readiness flag, always returns {"ready": true} (used by Docker/K8s readiness probes)
"""
from fastapi import APIRouter

from app.core.config import settings
from app.core.db import engine
from app.core.redis_client import get_redis
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    deps = {}

    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        deps["postgres"] = "ok"
    except Exception as exc:
        deps["postgres"] = f"error: {exc}"

    try:
        r = get_redis()
        await r.ping()
        deps["redis"] = "ok"
    except Exception as exc:
        deps["redis"] = f"error: {exc}"

    overall = "healthy" if all(v == "ok" for v in deps.values()) else "degraded"

    return HealthResponse(status=overall, service=settings.service_name, dependencies=deps)


@router.get("/ready")
async def ready():
    return {"ready": True}
