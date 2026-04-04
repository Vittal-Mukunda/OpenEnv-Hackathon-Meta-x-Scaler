---
title: OpenRnv
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

<h1 align="center">SchedulingOptEnv</h1>
<h3 align="center">A Markov Decision Environment for Training Autonomous<br>Scheduling Optimisation Agents</h3>

<p align="center"><em>Meta × Scaler — OpenEnv Hackathon Submission</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/framework-FastAPI-009688" alt="FastAPI">
  <img src="https://img.shields.io/badge/models-Pydantic%20v2-e92063" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/deploy-Docker%20%7C%20HF%20Spaces-yellow" alt="Docker | HF Spaces">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

---

## Abstract

We present **SchedulingOptEnv**, a real-world training environment for autonomous AI agents built upon the OpenEnv framework. The environment formalises combinatorial scheduling optimisation as a sequential decision problem, exposing agents to three progressively challenging sub-tasks: binary feasibility determination, multi-class constraint-violation classification, and full schedule repair. Each task is paired with a structured, differentiable reward function that provides dense, partial-progress signals rather than sparse binary outcomes. A 12-instance scheduling corpus covering five distinct constraint-violation classes, a FastAPI inference server, and a GPT-4o-mini baseline are included. The environment is deployable as a Docker container on Hugging Face Spaces with a single command.

---

## 1. Introduction

Combinatorial scheduling — the assignment of jobs to machines subject to resource, temporal, and precedence constraints — is a foundational problem in operations research, manufacturing, cloud computing, and logistics. Despite its industrial importance, existing benchmarks for evaluating AI agents on scheduling tasks are either purely offline (single-pass solution quality) or narrowly scoped to continuous optimisation rather than the constraint-satisfaction and repair workflow practised by human planners.

OpenEnv [1] provides an abstraction layer for building *interactive* environments where agents act, receive graded feedback, and improve across episodes. SchedulingOptEnv fills a gap by framing schedule analysis and repair as a Markov Decision Process (MDP) with:

- A well-defined **observation space** (JSON-encoded scheduling instance, task context, step counter)
- A structured **action space** (categorical labels or JSON repair schedules)
- A **multi-component reward function** that awards partial credit for structurally valid but suboptimal repairs
- Three **difficulty tiers** mirroring the cognitive complexity gradient faced by human schedulers

---

## 2. Environment Design

### 2.1 MDP Formulation

| Component | Definition |
|-----------|-----------|
| State *S* | Current scheduling instance, task type, step count, episode history |
| Observation *O* | `{schedule_instance: str (JSON), task_id, context, step_number}` |
| Action *A* | `{response: str, task_id: str}` |
| Reward *R* | Float ∈ [0.0, 1.0] from task-specific grader |
| Horizon *T* | Task-dependent: 3 / 5 / 8 steps |
| Terminal | *done* = True when *T* reached or *R* ≥ 0.95 |

### 2.2 Scheduling Instance Corpus

The environment ships with **12 curated scheduling instances** spanning five constraint-violation classes plus two fully feasible baselines. Instances are drawn from a task-aware pool: feasibility-check episodes see all 12, while classification and repair episodes see only the 10 infeasible instances.

| # | Feasible | Violation Class | Description |
|---|----------|----------------|-------------|
| 0 | No | `resource_overload` | J1 and J2 overlap on single-capacity machine M1 |
| 1 | No | `deadline_violation` | J1 starts late and finishes after hard deadline |
| 2 | No | `precedence_violation` | J2 starts before its predecessor J1 finishes |
| 3 | No | `availability_conflict` | J1 scheduled outside machine operating hours |
| 4 | No | `capacity_exceeded` | 3 concurrent jobs on capacity-2 machine |
| 5 | No | `resource_overload` | Pairwise overlap of J1 and J2 on capacity-1 machine |
| 6 | No | `deadline_violation` | Precedence chain forces J3 past hard deadline |
| 7 | No | `precedence_violation` | J3 starts before both predecessors complete |
| 8 | No | `availability_conflict` | J1 extends into machine maintenance window |
| 9 | No | `capacity_exceeded` | 4 concurrent jobs on capacity-3 machine |
| 10 | Yes | — | Fully feasible 3-job, 2-machine schedule |
| 11 | Yes | — | Fully feasible 5-job, 3-machine schedule with precedence |

---

## 3. Tasks

### Task 1 — Feasibility Check *(Easy)*

**Objective:** Given a JSON-encoded scheduling instance (jobs, machines, proposed assignments), determine whether the schedule satisfies all constraints.

**Action space:** `{"feasible", "infeasible"}`

**Grading function:**

```
R(a, g) = 1.0   if normalise(a) == ground_truth
          0.1   if a is non-empty but incorrect
          0.0   if a is empty
```

**Episode horizon:** 3 steps. **Target agent accuracy:** ~90%.

---

### Task 2 — Conflict Classification *(Medium)*

**Objective:** Identify the constraint violation present in an infeasible schedule from the closed vocabulary:
`{resource_overload, deadline_violation, precedence_violation, availability_conflict, capacity_exceeded}`

**Grading function:**

```
R(a, g) = 1.0   if a == ground_truth                             (exact)
          0.5   if a ∈ related_group(ground_truth)               (partial)
          0.1   if a ∈ valid_categories \ related_group(g)       (wrong family)
          0.0   if a ∉ valid_categories                          (unparseable)
```

where `related_groups = [{resource_overload, capacity_exceeded}, {deadline_violation, precedence_violation}]`.

**Episode horizon:** 5 steps. **Target agent accuracy:** ~60%.

---

### Task 3 — Schedule Repair *(Hard)*

**Objective:** Return a corrected schedule as a JSON object that resolves all constraint violations and minimises total makespan.

**Required JSON format:**
```json
{
  "assignments": [
    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
    {"job_id": "J2", "machine_id": "M1", "start_time": 4}
  ]
}
```

**Grading function (additive, max 1.0):**

```
R(a, g) = 0.2 × parseable_json(a)
        + 0.2 × valid_schema(a, g)
        + 0.4 × constraint_satisfaction_ratio(a, g)
        + 0.2 × optimality_score(makespan(a), makespan*(g))
```

where:
- `parseable_json(a)` — 1 if the response parses as valid JSON, else 0
- `valid_schema(a, g)` — 1 if all required fields are present and all jobs are assigned, else 0
- `constraint_satisfaction_ratio(a, g)` — fraction of four constraint categories satisfied:
  capacity, deadlines, precedence, availability (each worth 0.25)
- `optimality_score(m, m*)` — 1.0 if *m* ≤ 1.30·*m** ; 0.5 if *m* ≤ 1.60·*m** ; 0 otherwise

**Episode horizon:** 8 steps. **Target agent accuracy:** ~30%.

---

## 4. Server API

The environment is exposed over HTTP via a FastAPI server on port **7860** (Hugging Face Spaces default).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}` |
| `POST` | `/reset` | Begin new episode: `{"task_id": "feasibility_check"}` |
| `POST` | `/step` | Submit action: `{"response": "infeasible", "task_id": "feasibility_check"}` |
| `GET` | `/state` | Full internal state snapshot |
| `GET` | `/tasks` | Task catalogue with action schemas |
| `POST` | `/grader` | Direct grader invocation for offline evaluation |
| `GET` | `/baseline` | Trigger baseline inference; returns per-task scores |

---

## 5. Baseline

A standalone inference script (`baseline.py`) evaluates GPT-4o-mini on all three tasks. When `OPENAI_API_KEY` is not set, the script falls back to oracle mock responses, enabling offline verification of the grading pipeline without API access.

### 5.1 Baseline Scores (Mock / Oracle)

| Task | Instances | Average Score |
|------|-----------|--------------|
| Feasibility Check | 12 | 1.000 |
| Conflict Classification | 10 | 1.000 |
| Schedule Repair | 10 | 1.000 |
| **Overall** | | **1.000** |

---

## 6. Setup and Deployment

### 6.1 Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.11 |
| pip | ≥ 22.0 |
| Docker *(optional)* | ≥ 20.10 |
| Git | ≥ 2.30 |

### 6.2 Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/Vittal-Mukunda/OpenEnv-Hackathon-Meta-x-Scaler.git
cd OpenEnv-Hackathon-Meta-x-Scaler

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the server
uvicorn server:app --host 0.0.0.0 --port 7860

# 5. Verify the server is running
curl http://localhost:7860/health
# Expected: {"status":"ok"}
```

### 6.3 Docker Deployment

```bash
# Build the image
docker build -t scheduling-opt-env .

# Run the container
docker run -p 7860:7860 scheduling-opt-env

# Verify
curl http://localhost:7860/health
```

### 6.4 Hugging Face Spaces

Push this repository to a Hugging Face Space configured with the **Docker** SDK. The server listens on port 7860, which Spaces exposes automatically. No additional configuration is required.

### 6.5 Running the Baseline

```bash
# Without API key (uses oracle mock responses — scores 1.0 on all tasks)
python baseline.py

# With OpenAI API key (evaluates GPT-4o-mini)
export OPENAI_API_KEY=sk-...
python baseline.py
```

---

## 7. Example Interaction

```bash
# 1. Health check
curl http://localhost:7860/health

# 2. Start a feasibility-check episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "feasibility_check"}'

# 3. Submit a feasibility answer
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"response": "infeasible", "task_id": "feasibility_check"}'

# 4. Start a conflict-classification episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "conflict_classification"}'

# 5. Classify the violation
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"response": "resource_overload", "task_id": "conflict_classification"}'

# 6. Start a schedule-repair episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "schedule_repair"}'

# 7. Submit a repaired schedule
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "response": "{\"assignments\": [{\"job_id\": \"J1\", \"machine_id\": \"M1\", \"start_time\": 0}]}",
    "task_id": "schedule_repair"
  }'

# 8. Inspect environment state
curl http://localhost:7860/state

# 9. Invoke a grader directly
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{
    "action": {"response": "deadline_violation", "task_id": "conflict_classification"},
    "ground_truth": {"violation_type": "deadline_violation"}
  }'
```

---

## 8. Project Structure

```
.
├── openenv.yaml                  # OpenEnv metadata manifest
├── models.py                     # Pydantic v2 data models (Observation, Action, Reward)
├── environment.py                # SchedulingOptEnv core (reset / step / state + instance bank)
├── server.py                     # FastAPI HTTP server (7 endpoints)
├── baseline.py                   # GPT-4o-mini baseline with oracle fallback
├── Dockerfile                    # Container definition (python:3.11-slim, port 7860)
├── requirements.txt              # Python dependencies
├── tasks/
│   ├── __init__.py               # Task module exports
│   ├── task1_easy.py             # Feasibility check — episode runner + instance accessor
│   ├── task2_medium.py           # Conflict classification — episode runner + instance accessor
│   └── task3_hard.py             # Schedule repair — episode runner + instance accessor
└── graders/
    ├── __init__.py               # Grader exports (FeasibilityGrader, ConflictGrader, RepairGrader)
    ├── grader_detection.py       # Grader: feasibility (binary, synonym-aware)
    ├── grader_classification.py  # Grader: conflict classification (family-aware partial credit)
    └── grader_fix.py             # Grader: schedule repair (4-component additive reward)
```

---

## 9. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥ 0.104 | HTTP server framework |
| `uvicorn` | ≥ 0.24 | ASGI server |
| `pydantic` | ≥ 2.5 | Data validation and serialisation |
| `openai` | ≥ 1.6 | LLM baseline inference |
| `pyyaml` | ≥ 6.0 | YAML manifest parsing |
| `httpx` | ≥ 0.25 | Async HTTP client |

---

## 10. References

[1] OpenEnv Framework. *Building Real-World AI Agent Training Environments*. Meta × Scaler Hackathon, 2026.

[2] Pinedo, M. L. *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer, 2016.

[3] Garey, M. R., & Johnson, D. S. *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman, 1979.

[4] Zhang, C. et al. *Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning*. NeurIPS 2020.

[5] Kwon, Y.-D. et al. *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning*. NeurIPS 2020.
