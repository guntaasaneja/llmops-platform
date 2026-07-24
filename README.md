# LLMOps Platform — LLM Serving, Evaluation & Observability

A hands-on project demonstrating the core building blocks of running an LLM
application in production: serving requests through a REST API, caching
and rate-limiting, tracking prompt versions, evaluating output quality
(including a simple hallucination check), logging experiments, and
monitoring everything with Prometheus/Grafana — all containerized with
Docker and deployable to Kubernetes.

---

## 🧱 Architecture

```
                 ┌─────────────────────┐
                 │   GitHub Actions      │
                 │   CI/CD (lint→test→   │
                 │   build→deploy)        │
                 └──────────┬───────────┘
                            │
┌───────────────┐   ┌────────────────────┐   ┌──────────────────┐
│   API Caller    │──▶│   FastAPI Service    │──▶│  LLM Provider     │
│  (curl / client)│   │  (LLM Serving Layer) │   │  (Groq)  │
└───────────────┘   └────────┬───────────┘   └──────────────────┘
                            │
        ┌────────────────────┼────────────────────────┐
        ▼                     ▼                        ▼
┌───────────────┐   ┌──────────────────┐     ┌──────────────────┐
│  Redis Cache    │   │ PostgreSQL DB     │     │  MLflow Tracking  │
│ (rate-limit,    │   │ (prompts, runs,   │     │ (experiment logs   │
│  response cache)│   │  cost records)    │     │  per inference)    │
└───────────────┘   └──────────────────┘     └──────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│   Prometheus Metrics → Grafana Dashboard  │
│   (latency, cost, request volume)         │
└────────────────────────────────────────┘
```

## ⚙️ Tech Stack

| Category            | Technology                     |
|---------------------|---------------------------------|
| API / Serving        | FastAPI, Uvicorn                |
| Containerization      | Docker, Docker Compose           |
| Orchestration         | Kubernetes (Deployment, Service) |
| Experiment Tracking   | MLflow                          |
| Metrics               | Prometheus                      |
| Dashboards            | Grafana                         |
| Caching / Rate limit  | Redis                           |
| Persistence           | PostgreSQL                      |
| CI/CD                 | GitHub Actions                  |

## ✨ Features

- **LLM Serving** — a FastAPI endpoint that calls an LLM provider (OpenAI-compatible; works with Groq's free tier) with retry/backoff on transient failures.
- **Response Caching + Rate Limiting** — identical requests (same prompt/model/params) are served from Redis instead of re-calling the LLM; a simple per-client rate limiter protects against abuse.
- **Prompt Versioning** — prompt templates are stored in Postgres. Every edit creates a new version instead of overwriting, so you can view history and roll back.
- **Evaluation** — every completion is scored for relevance (does it address the prompt?), a simple hallucination check (word-overlap against a reference answer), and basic toxicity.
- **Experiment Tracking** — every inference call is logged to MLflow (params, tokens, latency, cost) so runs are comparable in the MLflow UI.
- **Cost Tracking** — token usage is converted into an estimated USD cost per request and exposed as a Prometheus metric.
- **Observability** — Prometheus scrapes `/metrics`; a pre-built Grafana dashboard shows latency, request volume, cost, and cache hit rate.
- **CI/CD** — a GitHub Actions pipeline: lint → test → build Docker image → deploy.
- **Containerized & Kubernetes-ready** — runs locally with one `docker-compose up`, with Kubernetes manifests included for deploying the same containers to a cluster.

## 🚀 Quickstart

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker-compose up --build
```

Services once running:
- API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5002
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## 🧪 Running tests

```bash
pytest tests/ -v
```

## 🔍 Try it out

```bash
# Run inference
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'

# Score a completion directly
curl -X POST http://localhost:8000/evaluation/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Capital of France?", "completion": "Paris.", "reference_answer": "Paris is the capital of France."}'

# Create a versioned prompt
curl -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"name": "greeting", "template": "Answer this: {input}", "description": "test prompt"}'
```

## ☸️ Kubernetes Deployment

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 📁 Project Structure

```
llmops-platform/
├── app/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── api/routes/              # inference, prompts, evaluation, health endpoints
│   ├── core/                    # config, db, redis clients
│   ├── services/                # llm_service, prompt_registry, cost_tracker, evaluation
│   ├── monitoring/               # Prometheus metrics
│   ├── models/                  # Pydantic schemas + SQLAlchemy models
│   └── mlflow_tracking/          # MLflow experiment logging
├── k8s/                         # Kubernetes manifests (deploy, service, configmap)
├── monitoring/                  # Prometheus + Grafana provisioning
├── tests/                       # pytest suite
├── .github/workflows/            # CI/CD pipeline
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 📊 Summary

- Built an **LLM serving API** with **FastAPI**, adding **Redis** response caching and rate limiting, and a **PostgreSQL**-backed **prompt versioning** system with rollback.
- Implemented an **evaluation pipeline** scoring relevance and a custom word-overlap **hallucination check**, with every inference run logged to **MLflow** for **experiment tracking**.
- Instrumented **Prometheus** metrics (latency, cost, request volume) with a **Grafana** dashboard for real-time monitoring.
- Containerized the full stack with **Docker** and **Docker Compose**; wrote **Kubernetes** manifests for deployment, plus a **GitHub Actions CI/CD** pipeline (lint, test, build, deploy).
