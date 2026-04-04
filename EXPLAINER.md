# SchedulingOptEnv
## A Markov Decision Environment for Training Autonomous Scheduling Optimisation Agents

**Meta × Scaler — OpenEnv Hackathon Submission**

**Team Vector** — Vittal Mukunda · Nikhilesh Nuthalapati · Stavan Rahul Khobare (Lead)

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Background and Motivation](#3-background-and-motivation)
4. [System Architecture](#4-system-architecture)
5. [Methodology](#5-methodology)
6. [The Scheduling Instance Corpus](#6-the-scheduling-instance-corpus)
7. [Task Definitions and Grading Functions](#7-task-definitions-and-grading-functions)
8. [Reward Design Philosophy](#8-reward-design-philosophy)
9. [Data Models](#9-data-models)
10. [API Specification](#10-api-specification)
11. [Inference and Baseline](#11-inference-and-baseline)
12. [Setup and Installation](#12-setup-and-installation)
13. [How to Get Results](#13-how-to-get-results)
14. [End-to-End Walkthrough](#14-end-to-end-walkthrough)
15. [Evaluation and Scoring](#15-evaluation-and-scoring)
16. [Project Structure](#16-project-structure)
17. [Dependencies](#17-dependencies)
18. [Glossary](#18-glossary)

---

## 1. Abstract

We present **SchedulingOptEnv**, a real-world AI agent training environment built on the OpenEnv framework for the Meta × Scaler OpenEnv Hackathon. The environment frames combinatorial scheduling optimisation as a sequential decision problem — a Markov Decision Process (MDP) — exposing agents to three progressively harder sub-tasks: binary feasibility determination, multi-class constraint-violation classification, and full schedule repair.

Each task is paired with a structured, differentiable reward function that provides dense, partial-progress signals rather than sparse binary outcomes. This design choice ensures that an AI agent always has a meaningful learning signal at every step, even when its answer is wrong — accelerating convergence during training.

The environment ships with:
- A **12-instance scheduling corpus** covering five distinct constraint-violation classes and two fully feasible baseline schedules
- A **FastAPI HTTP inference server** with 7 endpoints, exposing the full OpenEnv API contract
- A **standalone inference script** (`inference.py`) using the OpenAI client with configurable `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` environment variables
- A **GPT-4o-mini baseline** with oracle mock fallback for offline verification
- A **Docker container** deployable to Hugging Face Spaces with a single command

Oracle baseline achieves perfect scores (1.0) on all three tasks. The environment is fully spec-compliant with the OpenEnv standard and passes all automated pre-submission validation checks.

---

## 2. Introduction

### 2.1 What Is This Project?

In plain terms: **SchedulingOptEnv is a training gym where AI learns to detect, classify, and fix broken work schedules.**

Imagine a factory that needs to schedule 5 machines and 20 jobs. Some jobs can only run after other jobs finish. Some machines can only handle 2 jobs at once. Some machines are offline for maintenance from 3pm to 6pm. If someone hands the factory manager a schedule that violates any of these rules, the schedule is broken and production will fail.

This project creates a structured training playground where an AI agent can:
1. Look at a proposed schedule
2. Decide whether it is valid or broken
3. If broken, identify which type of rule was broken
4. Produce a corrected schedule that satisfies all constraints and minimises total time

The agent learns by trial and error: it receives a graded reward when it responds, and over thousands of practice rounds, it improves.

### 2.2 Why Scheduling?

Scheduling is one of the most practically important and computationally hard problems in computer science. It appears across nearly every industry:

| Industry | Scheduling Problem |
|----------|--------------------|
| Manufacturing | Assigning jobs to machines, managing shift constraints |
| Healthcare | Booking operating rooms, allocating staff and equipment |
| Aviation | Scheduling flights, crew rotations, gate assignments |
| Cloud computing | Allocating compute tasks to servers with capacity limits |
| Construction | Sequencing tasks with dependencies and resource limits |
| Logistics | Routing vehicles with time-window constraints |

Despite its industrial importance, prior benchmarks for evaluating AI agents on scheduling tasks were either:
- **Offline only** — agents see a problem once and produce a single answer, with no iterative improvement
- **Narrowly scoped** — focused purely on optimisation, not on the constraint-satisfaction and repair workflow that human planners actually use day-to-day

SchedulingOptEnv fills this gap. It creates an interactive, multi-step environment where an agent can reason, receive feedback, and refine its answers — mirroring the real cognitive workflow of a scheduling expert.

### 2.3 What Is OpenEnv?

OpenEnv is a framework specification by Meta and Hugging Face for building standardised, interactive AI agent training environments. It defines a common API contract — `step()`, `reset()`, `state()` — that any AI training system can connect to. By building SchedulingOptEnv on top of OpenEnv, any compatible AI agent or training loop can immediately start learning from our environment without any custom integration work.

---

## 3. Background and Motivation

### 3.1 The Scheduling Problem Formally

A scheduling instance consists of:

- A set of **jobs** `J = {J1, J2, ..., Jn}`, each with:
  - `duration` — how long the job takes to run
  - `deadline` — the latest time by which the job must be completed
  - `dependencies` — a list of job IDs that must complete before this job starts
  - `resource_req` — how much capacity this job consumes on its assigned machine

- A set of **machines** `M = {M1, M2, ..., Mm}`, each with:
  - `capacity` — how many jobs can run concurrently
  - `available_start` — the earliest time the machine is operational
  - `available_end` — the latest time the machine is operational

- A **proposed schedule** — a set of assignments `{(job_id, machine_id, start_time)}` specifying when and where each job runs

A schedule is **feasible** if and only if it satisfies all four constraint categories simultaneously:

| Constraint | Formal Definition |
|-----------|-------------------|
| **Capacity** | At every time `t`, the number of jobs concurrently running on machine `m` does not exceed `m.capacity` |
| **Deadline** | For every job `j`, `start_time(j) + duration(j) ≤ deadline(j)` |
| **Precedence** | For every job `j` and every predecessor `p ∈ dependencies(j)`: `start_time(j) ≥ start_time(p) + duration(p)` |
| **Availability** | For every job `j` on machine `m`: `start_time(j) ≥ m.available_start` and `start_time(j) + duration(j) ≤ m.available_end` |

A schedule is **infeasible** if it violates any one of these.

### 3.2 Why This Is Hard

Even checking whether a schedule is feasible is non-trivial for large instances. Repairing a broken schedule to be both feasible and optimal is NP-hard in general. Human planners develop intuition over years of experience. This project asks: can we create an environment rich enough for an AI to develop similar intuition through reinforcement learning?

### 3.3 The Gap in Existing Benchmarks

Existing scheduling benchmarks (e.g., JSP, FJSP) focus on finding optimal schedules from scratch. They do not model:
- The iterative detect → classify → repair workflow
- Partial-credit rewards for nearly-correct repairs
- Multi-step episodes where agents can refine answers
- Diverse constraint violation types as a classification target

SchedulingOptEnv addresses all of these gaps.

---

## 4. System Architecture

The system is organised into five logical layers, each building on the one below it.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — EXTERNAL CLIENTS                                             │
│  AI agents, training loops, researchers, the inference script           │
│  Communicate via HTTP JSON requests to port 7860                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  HTTP / JSON
┌────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 4 — API SERVER  (server.py)                                      │
│  FastAPI application with 7 REST endpoints                              │
│  Validates requests, routes to environment, serialises responses        │
│  Raises HTTP 400/422 for malformed input                                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  Python function calls
┌────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 3 — ENVIRONMENT  (environment.py)                                │
│  SchedulingOptEnv class — the core "game engine"                        │
│  Manages: episode lifecycle, instance selection, step counting,         │
│  termination logic, cumulative reward tracking, action history          │
│  Holds: INSTANCE_BANK (12 scheduling problems), grader singletons       │
└──────────────────┬──────────────────────────────────┬───────────────────┘
                   │                                  │
      ┌────────────▼──────────┐          ┌────────────▼──────────────────┐
      │  LAYER 2a — TASKS     │          │  LAYER 2b — GRADERS           │
      │  (tasks/ folder)      │          │  (graders/ folder)            │
      │                       │          │                               │
      │  task1_easy.py        │          │  grader_detection.py          │
      │  task2_medium.py      │          │    FeasibilityGrader          │
      │  task3_hard.py        │          │    (binary, synonym-aware)    │
      │                       │          │                               │
      │  Each task exposes:   │          │  grader_classification.py     │
      │  - episode runner     │          │    ConflictGrader             │
      │  - instance accessor  │          │    (family-aware partial)     │
      │  - task metadata      │          │                               │
      └────────────┬──────────┘          │  grader_fix.py               │
                   │                     │    RepairGrader               │
                   │                     │    (4-component additive)     │
                   │                     └────────────┬──────────────────┘
                   │                                  │
┌──────────────────▼──────────────────────────────────▼──────────────────┐
│  LAYER 1 — DATA MODELS  (models.py)                                     │
│  Pydantic v2 schemas: Observation, Action, Reward                       │
│  The shared type contract all layers use to communicate                 │
│  Enforces field types, constraints (score ∈ [0.0, 1.0]), and docs      │
└─────────────────────────────────────────────────────────────────────────┘

Additional components (alongside Layer 4):
  inference.py   — OpenAI-client inference script (API_BASE_URL / MODEL_NAME / HF_TOKEN)
  baseline.py    — GPT-4o-mini evaluation with oracle mock fallback
  openenv.yaml   — OpenEnv metadata manifest (machine-readable spec declaration)
  Dockerfile     — Container definition (python:3.11-slim, port 7860)
```

### 4.1 Component Responsibilities

**`models.py`** — Defines the three Pydantic v2 schemas that every other component imports:
- `Observation` — what the agent sees at each step (the schedule, task ID, instructions, step number)
- `Action` — what the agent submits (its text response + task ID)
- `Reward` — what the grader returns (float score + human-readable feedback string)

**`environment.py`** — The central engine. Contains:
- `INSTANCE_BANK` — the list of 12 scheduling instances with ground-truth answers
- `SchedulingOptEnv` class with `reset()`, `step()`, and `state()` methods
- Task-aware instance routing (Tasks 2 and 3 only see infeasible instances)
- Round-robin instance cycling to ensure all instances are covered during training
- Per-step contextual hints embedded in the `context` field of each Observation

**`server.py`** — FastAPI web application. Creates one shared `SchedulingOptEnv` instance and exposes it over 7 HTTP endpoints. Handles input validation and error responses.

**`graders/`** — Three independent grader classes, each implementing a `grade(action, ground_truth) → float` method. Graders are stateless between calls but store the last grading breakdown in `last_breakdown` for the environment to surface in the `info` dict.

**`tasks/`** — Thin wrappers that expose episode-running logic and instance pools for each task. These are used both by the environment and directly in the inference script.

**`inference.py`** — Standalone evaluation script. Reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables. Creates an OpenAI client pointed at the configured endpoint. Runs all 32 episodes (12 + 10 + 10) and emits `[START]`, `[STEP]`, and `[END]` structured JSON logs to stdout.

---

## 5. Methodology

### 5.1 MDP Formulation

The environment is formalised as a Markov Decision Process with the following components:

| MDP Component | Definition |
|--------------|-----------|
| **State** `S` | Current scheduling instance + task type + step count + episode history |
| **Observation** `O` | `{schedule_instance: str (JSON), task_id: str, context: str, step_number: int}` |
| **Action** `A` | `{response: str, task_id: str}` |
| **Reward** `R` | Float ∈ [0.0, 1.0] from the task-specific grader |
| **Horizon** `T` | 3 steps (Task 1) / 5 steps (Task 2) / 8 steps (Task 3) |
| **Terminal** | `done = True` when T is reached OR reward ≥ 0.95 |

The observation is always a JSON-encoded scheduling instance paired with a context string that instructs the agent on what to do. The action is always a text string — the agent's natural-language or structured-JSON response.

### 5.2 Episode Lifecycle

Every episode follows this exact sequence:

```
1. reset(task_id) is called
   ├── Validate task_id ∈ {feasibility_check, conflict_classification, schedule_repair}
   ├── Select instance pool for this task
   ├── Pick next instance via round-robin counter
   ├── Set step = 0, done = False, cumulative_reward = 0.0
   ├── Build initial context string (task instructions)
   └── Return Observation

2. Agent reads the Observation and formulates a response

3. step(action) is called
   ├── Increment step counter
   ├── Select correct grader based on task_id
   ├── Call grader.grade(action, instance_ground_truth)
   ├── Clamp reward to [0.0, 1.0]
   ├── Accumulate cumulative_reward
   ├── Log (step, action, reward) to history
   ├── Check termination: done = (step ≥ max_steps OR reward ≥ 0.95)
   └── Return (Observation, reward, done, info)

4. If done = False → repeat from step 2
   If done = True → episode ends; call reset() to start the next episode
```

### 5.3 Task-Aware Instance Routing

Not all 12 instances are appropriate for every task:

- **Task 1 (Feasibility Check)** — Uses all 12 instances (10 infeasible + 2 feasible), because the agent must learn to recognise both valid and invalid schedules
- **Task 2 (Conflict Classification)** — Uses only the 10 infeasible instances, because the task presupposes the schedule is broken
- **Task 3 (Schedule Repair)** — Uses only the 10 infeasible instances with known optimal repairs

### 5.4 Progressive Difficulty Design

The three tasks form a deliberate cognitive ladder:

```
Task 1 — Feasibility Check (EASY)
  ↓ Binary yes/no decision
  ↓ No structural reasoning required beyond spotting a single violation
  ↓ Target accuracy: ~90%

Task 2 — Conflict Classification (MEDIUM)
  ↓ Must identify WHICH of 5 violation categories is present
  ↓ Requires understanding of all four constraint types
  ↓ Target accuracy: ~60%

Task 3 — Schedule Repair (HARD)
  ↓ Must produce a syntactically valid, structurally correct, constraint-satisfying,
    near-optimal repaired schedule as JSON
  ↓ Requires full combinatorial reasoning
  ↓ Target accuracy: ~30%
```

This design ensures that agents can make measurable progress through the curriculum, and that the hardest task is genuinely challenging for frontier models.

---

## 6. The Scheduling Instance Corpus

The environment ships with 12 hand-crafted scheduling instances. Each instance is a Python dictionary with the following structure:

```python
{
    "instance": {               # exposed to the agent
        "problem_id": "P01",
        "jobs": [...],
        "machines": [...],
        "proposed_schedule": {"assignments": [...]}
    },
    "is_feasible": False,       # ground truth for Task 1
    "violation_type": "resource_overload",  # ground truth for Task 2
    "optimal_schedule": {"assignments": [...]},  # ground truth for Task 3
    "optimal_makespan": 7,      # used for optimality scoring in Task 3
    "description": "..."        # human-readable summary
}
```

### 6.1 Full Instance Catalogue

| # | Problem ID | Feasible | Violation Class | Key Constraint Broken |
|---|-----------|---------|----------------|----------------------|
| 0 | P01 | No | `resource_overload` | J1[0,4) and J2[2,5) overlap on M1 (capacity=1) |
| 1 | P02 | No | `deadline_violation` | J1 starts at t=5, finishes at t=10 > deadline=8 |
| 2 | P03 | No | `precedence_violation` | J2 starts at t=0 but depends on J1 which ends at t=8 |
| 3 | P04 | No | `availability_conflict` | J1 starts at t=5, before M1's window opens at t=8 |
| 4 | P05 | No | `capacity_exceeded` | 3 concurrent jobs on M1 with capacity=2 |
| 5 | P06 | No | `resource_overload` | J1 and J2 overlap on capacity-1 M1 (second instance) |
| 6 | P07 | No | `deadline_violation` | Precedence chain forces J3 past its hard deadline |
| 7 | P08 | No | `precedence_violation` | J3 starts before both J1 and J2 complete |
| 8 | P09 | No | `availability_conflict` | J1 extends into M1's maintenance window |
| 9 | P10 | No | `capacity_exceeded` | 4 concurrent jobs on M1 with capacity=3 |
| 10 | P11 | Yes | — | Fully feasible 3-job, 2-machine schedule |
| 11 | P12 | Yes | — | Fully feasible 5-job, 3-machine schedule with precedence |

### 6.2 Violation Class Distribution

| Class | Count | Description |
|-------|-------|-------------|
| `resource_overload` | 2 | Two jobs time-overlap on a single-capacity machine |
| `deadline_violation` | 2 | A job finishes after its required deadline |
| `precedence_violation` | 2 | A job starts before its predecessor completes |
| `availability_conflict` | 2 | A job is scheduled outside a machine's availability window |
| `capacity_exceeded` | 2 | Concurrent jobs exceed a machine's capacity limit |
| Feasible (no violation) | 2 | Used only in Task 1 |

Each violation class appears exactly twice, ensuring balanced representation and preventing class imbalance during training.

---

## 7. Task Definitions and Grading Functions

### 7.1 Task 1 — Feasibility Check (Easy)

**Objective:** Determine whether a proposed schedule satisfies all four constraint categories.

**Action space:** `{"feasible", "infeasible"}`

**Episode horizon:** 3 steps

**Grader: `FeasibilityGrader`**

The grader normalises common synonyms before scoring, so agents that reply with natural language equivalents still receive full credit:

| Synonyms treated as "feasible" | Synonyms treated as "infeasible" |
|-------------------------------|----------------------------------|
| feasible, valid, correct, satisfiable, yes, ok, pass | infeasible, invalid, incorrect, unsatisfiable, no, violated, conflict, fail, impossible, broken |

**Scoring function:**

```
R(action, ground_truth) =
    1.0   if normalise(action.response) == ground_truth.is_feasible
    0.1   if response is non-empty but cannot be parsed OR is wrong
    0.0   if response is empty
```

The `0.1` score for a wrong-but-present answer is deliberate. It ensures the agent always has a gradient signal — even a completely wrong answer tells the training system "the agent tried; adjust its policy."

---

### 7.2 Task 2 — Conflict Classification (Medium)

**Objective:** Identify the constraint violation type present in an infeasible schedule from the closed vocabulary of five classes.

**Action space:** `{resource_overload, deadline_violation, precedence_violation, availability_conflict, capacity_exceeded}`

**Episode horizon:** 5 steps

**Grader: `ConflictGrader`**

The grader is aware of semantic "constraint families" — groups of violation types that are conceptually related. This enables partial credit for answers that are wrong but not completely off-base.

**Constraint families:**
- **Resource-limit family:** `resource_overload` and `capacity_exceeded` — both concern concurrent job count on a machine
- **Temporal-ordering family:** `deadline_violation` and `precedence_violation` — both concern job timing and sequencing
- **Standalone:** `availability_conflict` — concerns machine operational windows

**Scoring function:**

```
R(action, ground_truth) =
    1.0   if action.response == ground_truth.violation_type             (exact match)
    0.5   if action.response is in the same family as ground_truth       (partial credit)
    0.1   if action.response is a valid category but wrong family        (attempted)
    0.0   if action.response is not a valid category                     (unparseable)
```

The grader also normalises spacing and dashes to underscores, so `"deadline violation"` and `"deadline-violation"` both map to `"deadline_violation"` before scoring.

---

### 7.3 Task 3 — Schedule Repair (Hard)

**Objective:** Return a corrected schedule as a JSON object that resolves all constraint violations and minimises total makespan.

**Required output format:**

```json
{
  "assignments": [
    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
    {"job_id": "J2", "machine_id": "M1", "start_time": 4},
    {"job_id": "J3", "machine_id": "M2", "start_time": 0}
  ]
}
```

**Episode horizon:** 8 steps

**Grader: `RepairGrader`**

The grader evaluates four independent components and sums them additively to a maximum of 1.0:

**Component 1 — JSON Parseability (20% of score)**

```
0.20   if the response parses as a valid JSON object
0.00   if the response is not valid JSON (no partial credit at this level)
```

The parser uses three strategies in sequence:
1. Direct `json.loads()` — handles pure JSON responses
2. Strip markdown code fences (```` ``` ````) then parse — handles LLM wrapping
3. Brace-counting to extract the outermost `{...}` block — handles prose-wrapped JSON

**Component 2 — Schema Validity (20% of score)**

```
0.20   if the JSON contains "assignments" list, every assignment has
       {job_id, machine_id, start_time}, start_time ≥ 0, and every job
       from the instance appears exactly once
0.00   otherwise
```

**Component 3 — Constraint Satisfaction (40% of score)**

Checks all four constraint categories independently. Each is worth 10% of the total score (0.10 points):

```
capacity_ok     — no machine has more concurrent jobs than its capacity
deadlines_ok    — every job finishes at or before its deadline
precedence_ok   — every job starts after all its predecessors finish
availability_ok — every job runs within its machine's operational window

score += 0.40 × (number of satisfied categories / 4)
```

**Component 4 — Makespan Optimality (20% of score)**

```
0.20   if makespan(response) ≤ optimal_makespan × 1.30   (within 30% of optimal)
0.10   if makespan(response) ≤ optimal_makespan × 1.60   (within 60% of optimal)
0.00   if makespan(response) > optimal_makespan × 1.60   (too slow)
```

Where `makespan` is defined as the maximum finish time across all jobs: `max(start_time(j) + duration(j)) ∀j`.

**Full scoring formula:**

```
R = 0.20 × parseable_json(response)
  + 0.20 × valid_schema(response)
  + 0.40 × (satisfied_constraints / 4)
  + 0.20 × optimality_score(makespan, optimal_makespan)
```

---

## 8. Reward Design Philosophy

### 8.1 Dense vs Sparse Rewards

A key design principle in this environment is the use of **dense rewards** — reward signals that give partial credit throughout the episode, not just at the end.

**Sparse reward (bad for learning):**
```
score = 1.0 if the answer is perfect
score = 0.0 otherwise
```

This gives the agent almost no information. If the answer is wrong, the agent doesn't know if it was slightly wrong or completely wrong.

**Dense reward (what this environment uses):**
```
score = sum of partial credits for each correct sub-component
```

This tells the agent exactly which parts of its answer were right and which were wrong, enabling it to make targeted improvements in subsequent steps.

### 8.2 Why Partial Credit Matters

Consider a Schedule Repair attempt that produces syntactically valid JSON, covers all jobs, and fixes 3 out of 4 constraints, but is 50% slower than optimal:

```
score = 0.20 (JSON valid)
      + 0.20 (schema valid)
      + 0.30 (3/4 constraints: 0.40 × 0.75)
      + 0.10 (makespan within 60% of optimal)
      = 0.80
```

The agent knows it is close. It knows it missed one constraint and its solution is a bit slow. It can target those specific issues in the next step.

### 8.3 The 0.1 Floor for Task 1 and Task 2

Even an incorrect answer receives a non-zero score (0.1) as long as the response is non-empty and parseable. This is intentional: it prevents the agent from learning to produce empty responses as a strategy to avoid penalties, and it keeps the gradient non-zero for wrong-but-present answers.

---

## 9. Data Models

All inter-component communication uses three Pydantic v2 schemas defined in `models.py`:

### Observation
```python
class Observation(BaseModel):
    schedule_instance: str   # JSON-encoded scheduling problem
    task_id: str             # "feasibility_check" | "conflict_classification" | "schedule_repair"
    context: str             # Natural-language instructions for the current step
    step_number: int         # Current step (0-indexed, ge=0)
```

### Action
```python
class Action(BaseModel):
    response: str    # Agent's text answer (e.g., "infeasible", "resource_overload", or JSON)
    task_id: str     # Must match the current episode's task_id
```

### Reward
```python
class Reward(BaseModel):
    score: float     # Grading result, enforced ∈ [0.0, 1.0]
    feedback: str    # Human-readable explanation of the score
```

---

## 10. API Specification

The environment is exposed as a RESTful HTTP API via FastAPI, running on port **7860** (the Hugging Face Spaces default).

### Endpoints

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|-------------|----------|
| `GET` | `/health` | Liveness probe | None | `{"status": "ok"}` |
| `POST` | `/reset` | Start new episode | `{"task_id": "..."}` | `Observation` |
| `POST` | `/step` | Submit action | `{"response": "...", "task_id": "..."}` | `StepResponse` |
| `GET` | `/state` | Internal state snapshot | None | Full state dict |
| `GET` | `/tasks` | Task catalogue | None | Array of task descriptions |
| `POST` | `/grader` | Direct grader invocation | `{action, ground_truth}` | `{"score": float}` |
| `GET` | `/baseline` | Run GPT-4o-mini baseline | None | Per-task scores |

### StepResponse Schema

```json
{
  "observation": { "schedule_instance": "...", "task_id": "...", "context": "...", "step_number": 1 },
  "reward": 1.0,
  "done": true,
  "info": {
    "step_reward": 1.0,
    "cumulative_reward": 1.0,
    "steps_remaining": 2,
    "instance_description": "...",
    "grading_breakdown": { ... }
  }
}
```

The `info.grading_breakdown` dict exposes the full internal grading decision — predicted vs expected, per-constraint pass/fail flags, makespan ratio — enabling training loops to inspect the decision without parsing the float reward.

---

## 11. Inference and Baseline

### 11.1 inference.py

The primary evaluation script. Uses three environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `API_BASE_URL` | Base URL for the OpenAI-compatible API | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | API key or Hugging Face token | `hf_...` or `sk-...` |

**Log format:**

Every episode emits three structured JSON log lines to stdout:

```
[START] {"task_id": "feasibility_check", "instance_id": 0}
[STEP]  {"task_id": "feasibility_check", "instance_id": 0, "step": 1, "action": "infeasible", "reward": 1.0, "done": true, "feedback": "Correct."}
[END]   {"task_id": "feasibility_check", "instance_id": 0, "final_reward": 1.0}
```

Final summary line:

```json
{"event": "eval_end", "summary": {"feasibility_check": {"average_score": 1.0, "num_instances": 12}, "conflict_classification": {"average_score": 1.0, "num_instances": 10}, "schedule_repair": {"average_score": 1.0, "num_instances": 10}, "overall_average": 1.0}}
```

**Oracle fallback:** When `HF_TOKEN` is not set, the script falls back to deterministic mock responses (the ground-truth answers) and scores 1.0 on all tasks. This enables offline verification of the grading pipeline without any API access.

### 11.2 baseline.py

An alternative evaluation script (`baseline.py`) that calls the graders directly (without HTTP) and supports the same GPT-4o-mini / oracle-mock pattern. Useful for rapid local testing.

### 11.3 Baseline Scores

| Mode | Task 1 | Task 2 | Task 3 | Overall |
|------|--------|--------|--------|---------|
| Oracle mock (no API key) | 1.000 | 1.000 | 1.000 | 1.000 |
| GPT-4o-mini (with API key) | ~0.90 | ~0.60 | ~0.30 | ~0.60 |

The mock oracle achieves perfect scores by design — it is used to verify the grading pipeline, not to claim AI performance. The GPT-4o-mini scores reflect realistic expectations based on the designed difficulty levels.

---

## 12. Setup and Installation

### 12.1 Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.11 |
| pip | ≥ 22.0 |
| Docker Desktop *(for container testing)* | ≥ 20.10 |
| Git | ≥ 2.30 |

### 12.2 Local Installation

```bash
# Step 1 — Clone the repository
git clone https://github.com/Vittal-Mukunda/OpenEnv-Hackathon-Meta-x-Scaler.git
cd OpenEnv-Hackathon-Meta-x-Scaler

# Step 2 — Create a virtual environment (strongly recommended)
python -m venv .venv

# Activate it:
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat

# Step 3 — Install dependencies
pip install -r requirements.txt

# Step 4 — Start the server
uvicorn server:app --host 0.0.0.0 --port 7860

# Step 5 — Verify the server is alive (new terminal)
curl http://localhost:7860/health
# Expected: {"status":"ok"}
```

### 12.3 Docker Deployment

```bash
# Step 1 — Open Docker Desktop and wait for it to fully start

# Step 2 — Build the image (takes 1-2 minutes first time)
docker build -t scheduling-env .

# Step 3 — Run the container
docker run -p 7860:7860 scheduling-env

# Step 4 — Verify in a new terminal
curl http://localhost:7860/health
# Expected: {"status":"ok"}

# Step 5 — Stop the container
# Press Ctrl+C in the terminal running Docker
```

### 12.4 Hugging Face Spaces Deployment

1. Create a new Hugging Face Space at huggingface.co/new-space
2. Select **Docker** as the SDK
3. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space master
   ```
4. The Space will auto-detect the Dockerfile, build, and deploy. Port 7860 is exposed automatically.

---

## 13. How to Get Results

### 13.1 Quick Test — Oracle Mock (No API Key Needed)

This runs the full evaluation pipeline with deterministic oracle answers. Use this to verify everything is working correctly.

```bash
python inference.py
```

**Expected output:**

```
{"event": "eval_start", "mode": "oracle mock", "model": "gpt-4o-mini"}
[START] {"task_id": "feasibility_check", "instance_id": 0}
[STEP]  {"task_id": "feasibility_check", "instance_id": 0, "step": 1, "action": "infeasible", "reward": 1.0, "done": true, "feedback": "Correct."}
[END]   {"task_id": "feasibility_check", "instance_id": 0, "final_reward": 1.0}
... (32 episodes total) ...
{"event": "eval_end", "summary": {"feasibility_check": {"average_score": 1.0, "num_instances": 12}, "conflict_classification": {"average_score": 1.0, "num_instances": 10}, "schedule_repair": {"average_score": 1.0, "num_instances": 10}, "overall_average": 1.0}}
```

**What the output means:**
- Each `[START]` line: a new episode begins on a specific instance
- Each `[STEP]` line: the agent submitted an answer; shows what it said and what score it received
- Each `[END]` line: the episode is over; shows the final reward
- The last line: aggregated scores per task and overall

### 13.2 Real LLM Evaluation (With API Key)

```bash
# Using OpenAI
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-your-openai-key-here
python inference.py

# Using Hugging Face Inference API
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export HF_TOKEN=hf_your-token-here
python inference.py
```

### 13.3 Testing Individual Endpoints

With the server running (`uvicorn server:app --port 7860`):

```bash
# Health check
curl http://localhost:7860/health

# Start a feasibility episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "feasibility_check"}'

# Submit an answer
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"response": "infeasible", "task_id": "feasibility_check"}'

# Start a classification episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "conflict_classification"}'

# Submit classification
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"response": "resource_overload", "task_id": "conflict_classification"}'

# Start a repair episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "schedule_repair"}'

# Submit a repaired schedule
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"response": "{\"assignments\": [{\"job_id\": \"J1\", \"machine_id\": \"M1\", \"start_time\": 0}, {\"job_id\": \"J2\", \"machine_id\": \"M1\", \"start_time\": 4}]}", "task_id": "schedule_repair"}'

# Check full internal state
curl http://localhost:7860/state

# Grade an action directly (without going through an episode)
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"action": {"response": "deadline_violation", "task_id": "conflict_classification"}, "ground_truth": {"violation_type": "deadline_violation"}}'
```

### 13.4 Interpreting the Score

| Score Range | Meaning |
|------------|---------|
| 1.0 | Perfect answer |
| 0.8 – 0.99 | Excellent — nearly correct, minor issue |
| 0.5 – 0.79 | Good — correct constraint family or partial repair |
| 0.2 – 0.49 | Partial — JSON parsed, some constraints fixed |
| 0.1 – 0.19 | Attempted — something was submitted, but mostly wrong |
| 0.0 | Empty response or completely unparseable |

---

## 14. End-to-End Walkthrough

### Full Example: Task 3 Schedule Repair

This walkthrough shows exactly what happens when an agent attempts to repair instance P01 (resource_overload: J1[0,4) and J2[2,5) overlap on M1 capacity=1).

**Step 1 — Reset**

```
POST /reset
{"task_id": "schedule_repair"}
```

The environment:
1. Validates task_id = "schedule_repair"
2. Selects the infeasible instance pool (10 instances)
3. Picks instance 0 (P01) via round-robin
4. Sets step=0, done=False
5. Returns Observation:

```json
{
  "schedule_instance": "{\"problem_id\": \"P01\", \"jobs\": [{\"id\": \"J1\", \"duration\": 4, ...}, ...], \"proposed_schedule\": {\"assignments\": [{\"job_id\": \"J1\", \"start_time\": 0}, {\"job_id\": \"J2\", \"start_time\": 2}]}}",
  "task_id": "schedule_repair",
  "context": "The proposed_schedule is infeasible. Return ONLY a JSON object with key \"assignments\"...",
  "step_number": 0
}
```

**Step 2 — Agent submits broken JSON**

```
POST /step
{"response": "Here is my repair: {bad json", "task_id": "schedule_repair"}
```

RepairGrader tries all 3 parse strategies. All fail. Returns:
- `json_parseable: false` → score = 0.0
- `done: false` (step 1/8)

**Step 3 — Agent submits valid JSON, wrong schema**

```
POST /step
{"response": "{\"jobs\": [...]}", "task_id": "schedule_repair"}
```

RepairGrader: JSON parses (0.20), but no "assignments" key (schema fails). Returns:
- score = 0.20
- `done: false` (step 2/8)

**Step 4 — Agent submits correct schema, partial constraints**

```json
{"assignments": [{"job_id": "J1", "machine_id": "M1", "start_time": 0}, {"job_id": "J2", "machine_id": "M1", "start_time": 2}, {"job_id": "J3", "machine_id": "M2", "start_time": 0}]}
```

Grader checks:
- JSON: ✅ (0.20)
- Schema: ✅ all jobs present (0.20)
- Capacity: ❌ J1[0,4) and J2[2,5) still overlap on M1
- Deadlines: ✅
- Precedence: ✅
- Availability: ✅
- Constraints: 3/4 = 0.30
- Makespan: 7 vs optimal 7 → ratio 1.0 → 0.20

Score = 0.20 + 0.20 + 0.30 + 0.20 = **0.90**

**Step 5 — Agent submits optimal repair**

```json
{"assignments": [{"job_id": "J1", "machine_id": "M1", "start_time": 0}, {"job_id": "J2", "machine_id": "M1", "start_time": 4}, {"job_id": "J3", "machine_id": "M2", "start_time": 0}]}
```

Grader checks:
- JSON: ✅ (0.20)
- Schema: ✅ (0.20)
- Capacity: ✅ J1[0,4), J2[4,7) no overlap
- Deadlines: ✅
- Precedence: ✅
- Availability: ✅
- Constraints: 4/4 = 0.40
- Makespan: 7 vs optimal 7 → ratio 1.0 → 0.20

Score = **1.00**. `done: true`. Episode ends.

---

## 15. Evaluation and Scoring

### 15.1 Pre-Submission Checklist

| Check | Pass Condition |
|-------|---------------|
| HF Space deploys | `GET /health` returns 200 and `{"status":"ok"}` |
| OpenEnv spec compliance | `openenv.yaml` valid; typed models; `step()`/`reset()`/`state()` respond correctly |
| Dockerfile builds | `docker build` completes without errors |
| Baseline reproduces | `python inference.py` completes without error and produces scores |
| 3+ tasks with graders | All three tasks return scores ∈ [0.0, 1.0] |

### 15.2 Hackathon Scoring Breakdown

| Criterion | Weight | This Project |
|-----------|--------|-------------|
| Real-world utility | 30% | Industrial scheduling (manufacturing, cloud, healthcare) |
| Task & grader quality | 25% | 3 tasks, difficulty range easy→hard, deterministic graders |
| Environment design | 20% | Dense rewards, clean state, sensible episode boundaries |
| Code quality & spec compliance | 15% | Typed models, documented, Docker works, spec compliant |
| Creativity & novelty | 10% | Scheduling is rare in OpenEnv; multi-component repair grader is novel |

---

## 16. Project Structure

```
.
├── inference.py                  # Primary inference script (API_BASE_URL/MODEL_NAME/HF_TOKEN)
├── baseline.py                   # GPT-4o-mini baseline with oracle fallback
├── server.py                     # FastAPI HTTP server (7 endpoints, port 7860)
├── environment.py                # SchedulingOptEnv core + INSTANCE_BANK (12 problems)
├── models.py                     # Pydantic v2 data models (Observation, Action, Reward)
├── openenv.yaml                  # OpenEnv metadata manifest
├── Dockerfile                    # Container definition (python:3.11-slim, port 7860)
├── requirements.txt              # Python dependencies
├── tasks/
│   ├── __init__.py               # Task module exports
│   ├── task1_easy.py             # Feasibility check — episode runner + instance accessor
│   ├── task2_medium.py           # Conflict classification — episode runner + instance accessor
│   └── task3_hard.py             # Schedule repair — episode runner + instance accessor
└── graders/
    ├── __init__.py               # Grader exports
    ├── grader_detection.py       # FeasibilityGrader (binary, synonym-aware)
    ├── grader_classification.py  # ConflictGrader (family-aware partial credit)
    └── grader_fix.py             # RepairGrader (4-component additive reward)
```

---

## 17. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥ 0.104 | HTTP server framework |
| `uvicorn` | ≥ 0.24 | ASGI server (runs FastAPI) |
| `pydantic` | ≥ 2.5 | Data validation and serialisation |
| `openai` | ≥ 1.6 | LLM inference (used in inference.py and baseline.py) |
| `pyyaml` | ≥ 6.0 | YAML manifest parsing (openenv.yaml) |
| `httpx` | ≥ 0.25 | Async HTTP client (used internally by FastAPI) |

All dependencies are intentionally lightweight and ship in `python:3.11-slim` with no additional system packages required.

---

## 18. Glossary

| Term | Definition |
|------|-----------|
| **MDP** | Markov Decision Process — a formal framework for sequential decision problems |
| **Episode** | One complete round from `reset()` to terminal state |
| **Horizon** | Maximum number of steps allowed per episode |
| **Terminal** | State where `done = True` — episode must end |
| **Dense reward** | A reward signal that provides partial credit at every step, not just at the end |
| **Sparse reward** | A reward signal that is non-zero only at the final step (bad for learning) |
| **Feasible** | A schedule that satisfies all four constraint categories |
| **Infeasible** | A schedule that violates at least one constraint |
| **Makespan** | The total time from the start of the first job to the finish of the last job |
| **Optimal makespan** | The minimum achievable makespan for a given set of jobs and machines |
| **Instance** | One specific scheduling problem (jobs + machines + proposed schedule + ground truth) |
| **Grader** | A Python class that scores an agent's action against ground truth |
| **OpenEnv** | The framework specification by Meta/HuggingFace that this environment implements |
| **Constraint family** | A group of semantically related violation types used for partial-credit scoring |
| **Oracle** | A mock agent that always produces the ground-truth answer — used to verify the grading pipeline |

---

*Built for the Meta × Scaler OpenEnv Hackathon · April 2026 · MIT License*
