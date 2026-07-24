"""
SQLAlchemy ORM models — persistence layer for prompts, inference runs,
evaluation results, and cost records.
These define what data looks like in PostgreSQL — actual tables
"""
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Prompt(Base):
    """Versioned prompt template. Each edit creates a new version row so
    prior versions remain available for rollback / A-B comparison."""

    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    template: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    runs: Mapped[list["InferenceRun"]] = relationship(back_populates="prompt")


class InferenceRun(Base):
    """One record per LLM inference call — the audit trail for serving,
    cost tracking, and latency monitoring."""

    __tablename__ = "inference_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prompt_id: Mapped[int | None] = mapped_column(ForeignKey("prompts.id"), nullable=True)
    model: Mapped[str] = mapped_column(String(128))
    prompt_text: Mapped[str] = mapped_column(Text)
    completion_text: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prompt: Mapped["Prompt | None"] = relationship(back_populates="runs")
    evaluation: Mapped["EvaluationRecord | None"] = relationship(back_populates="run", uselist=False)


class EvaluationRecord(Base):
    """Automated evaluation results for a given inference run."""

    __tablename__ = "evaluation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("inference_runs.id"))
    relevance_score: Mapped[float] = mapped_column(Float)
    correctness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucination_score: Mapped[float] = mapped_column(Float)
    toxicity_score: Mapped[float] = mapped_column(Float)
    overall_pass: Mapped[bool] = mapped_column(Boolean)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    run: Mapped["InferenceRun"] = relationship(back_populates="evaluation")
