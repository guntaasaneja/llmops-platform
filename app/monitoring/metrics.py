"""
Prometheus metrics for the LLMOps platform.

Exposed at /metrics and scraped by Prometheus (see monitoring/prometheus.yml),
visualized in Grafana (see monitoring/grafana/dashboards).
"""
from prometheus_client import Counter, Gauge, Histogram

# --- Latency monitoring ---
INFERENCE_LATENCY = Histogram(
    "llm_inference_latency_seconds",
    "End-to-end LLM inference latency",
    ["model", "cached"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
)

# --- Request volume / errors ---
INFERENCE_REQUESTS_TOTAL = Counter(
    "llm_inference_requests_total", "Total inference requests", ["model", "status"]
)

# --- Cost tracking ---
INFERENCE_COST_USD_TOTAL = Counter(
    "llm_inference_cost_usd_total", "Cumulative estimated USD cost of LLM calls", ["model"]
)
TOKENS_USED_TOTAL = Counter(
    "llm_tokens_used_total", "Total tokens consumed", ["model", "token_type"]
)

# --- Evaluation / hallucination ---
HALLUCINATION_SCORE = Histogram(
    "llm_hallucination_score",
    "Hallucination score distribution (0=grounded, 1=hallucinated)",
    ["model"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
EVAL_PASS_RATE = Gauge("llm_eval_pass_rate", "Rolling evaluation pass rate", ["model"])

# --- Cache efficacy ---
CACHE_HITS_TOTAL = Counter("llm_cache_hits_total", "Response cache hits")
CACHE_MISSES_TOTAL = Counter("llm_cache_misses_total", "Response cache misses")

# --- System / deployment ---
MODEL_VERSION_INFO = Gauge(
    "llm_model_deployment_info", "Currently deployed model version (1=active)", ["model", "channel"]
)
