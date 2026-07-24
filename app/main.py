"""
FastAPI application entrypoint.

Wires together: API routers, Prometheus /metrics endpoint, DB startup, and
structured logging.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.api.routes import evaluation, health, inference, prompts
from app.core.config import settings
from app.core.db import init_db

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(settings.service_name)

app = FastAPI(
    title="LLMOps Platform",
    description="LLM serving, evaluation, and observability platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(inference.router)
app.include_router(prompts.router)
app.include_router(evaluation.router)


@app.on_event("startup")
async def on_startup():
    logger.info("Starting %s in %s mode", settings.service_name, settings.app_env)
    await init_db()


@app.get("/metrics")
async def metrics():
    """Scraped by Prometheus (see monitoring/prometheus.yml)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {
        "service": settings.service_name,
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics",
    }
