"""
MLflow integration — logs every inference + evaluation as an MLflow run so
prompt/model iterations are fully reproducible and comparable in the MLflow UI.
"""
import mlflow

from app.core.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

try:
    mlflow.set_experiment(settings.mlflow_experiment_name)
except Exception:
    # MLflow server not reachable (e.g. running app tests without the
    # docker-compose stack) — logging calls below become no-ops.
    pass


def log_inference_run(
    model: str,
    prompt: str,
    completion: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    cost_usd: float,
    hallucination_score: float | None = None,
) -> str | None:
    """Logs a single inference call as an MLflow run. Returns the run_id,
    or None if MLflow isn't reachable (fails soft, never blocks serving)."""
    try:
        with mlflow.start_run(run_name=f"inference-{model}") as run:
            mlflow.log_params(
                {
                    "model": model,
                    "prompt_preview": prompt[:200],
                }
            )
            mlflow.log_metrics(
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    **({"hallucination_score": hallucination_score} if hallucination_score is not None else {}),
                }
            )
            mlflow.log_text(completion, "completion.txt")
            return run.info.run_id
    except Exception:
        return None


def log_evaluation_batch(run_name: str, metrics: dict, params: dict | None = None) -> str | None:
    """Logs an offline batch-evaluation run (see app/evaluation/pipeline.py)."""
    try:
        with mlflow.start_run(run_name=run_name) as run:
            if params:
                mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            return run.info.run_id
    except Exception:
        return None
