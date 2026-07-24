"""
Pydantic schemas: request/response contracts for the API layer.
These define what data looks like going in and out of your API (request/response validation)
schemas.py = the API's view of data (what a client sends/receives)
db_models.py = the database's view of data (what's actually stored)
"""
from datetime import datetime

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="User prompt / input text")
    model: str | None = Field(default=None, description="Override default model")
    prompt_version: str | None = Field(default=None, description="Pinned prompt template version")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=8192)
    reference_answer: str | None = Field(
        default=None, description="Optional ground truth used for hallucination scoring"
    )
    stream: bool = False


class InferenceResponse(BaseModel):
    request_id: str
    completion: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    cached: bool = False
    hallucination_score: float | None = None
    mlflow_run_id: str | None = None


class PromptCreateRequest(BaseModel):
    name: str
    template: str
    description: str | None = None
    tags: list[str] = []


class PromptResponse(BaseModel):
    id: int
    name: str
    version: int
    template: str
    description: str | None
    tags: list[str]
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class EvaluationRequest(BaseModel):
    prompt: str
    completion: str
    reference_answer: str | None = None
    model: str | None = None


class EvaluationResult(BaseModel):
    relevance_score: float
    hallucination_score: float
    toxicity_score: float
    overall_pass: bool
    details: dict


class HealthResponse(BaseModel):
    status: str
    service: str
    dependencies: dict[str, str]
