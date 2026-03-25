# CodeReviewEnv: A Structured Environment for Training Autonomous Code Review Agents

**Meta × Scaler OpenEnv Hackathon Submission**
Vittal Mukunda · 2026

---

## Abstract

We present **CodeReviewEnv**, a real-world training environment for autonomous AI agents built on the OpenEnv framework. The environment operationalises the software engineering task of code review as a sequential decision problem, exposing agents to three progressively difficult sub-tasks: binary bug detection, multi-class bug classification, and full code repair. Each task is accompanied by a structured grading function that provides dense, partial-progress reward signals rather than sparse binary outcomes. A 12-snippet Python code corpus, a FastAPI inference server, and a GPT-4o-mini baseline are included. The environment is deployable as a Docker container on Hugging Face Spaces with a single command.

---

## 1. Introduction

Automated code review is among the most economically valuable applications of large language models. Human code review consumes a significant fraction of engineering time in production software teams, yet existing benchmarks for evaluating code-reviewing AI agents are either purely static (single-shot pass/fail) or narrowly scoped to code generation rather than review. OpenEnv [1] provides an abstraction layer for building _interactive_ environments where agents act, receive graded feedback, and improve over episodes.

CodeReviewEnv fills a gap by framing code review as a Markov Decision Process (MDP) with:

- A well-defined **observation space** (code snippet, task context, step counter)
- A structured **action space** (natural-language responses or code patches)
- A **multi-component reward function** that awards partial credit for syntactically valid but semantically incomplete solutions
- Three **difficulty tiers** reflecting real-world cognitive load gradients in code review

---

## 2. Environment Design

### 2.1 MDP Formulation

| Component | Definition |
|-----------|-----------|
| State *S* | Current code snippet, task type, step count, episode history |
| Observation *O* | `{code_snippet, task_id, context, step_number}` |
| Action *A* | `{response: str, task_id: str}` |
| Reward *R* | Float ∈ [0.0, 1.0] from task-specific grader |
| Horizon *T* | Task-dependent: 3 / 5 / 8 steps |
| Terminal | *done* = True when *T* reached or *R* ≥ 0.95 |

### 2.2 Code Corpus

The environment ships with 12 curated Python snippets spanning a representative distribution of real-world defect classes:

| # | Defect Class | Snippet Description |
|---|--------------|---------------------|
| 0 | Logic error | Off-by-one in loop range |
| 1 | Security flaw | SQL injection via f-string |
| 2 | Null reference | IndexError on empty list |
| 3 | Logic error | Infinite loop (missing decrement) |
| 4 | Logic error | Wrong variable in f-string |
| 5 | Logic error | Missing return statement |
| 6 | Logic error | Type mismatch (int + str) |
| 7 | Security flaw | Insecure randomness (`random` vs `secrets`) |
| 8 | Performance issue | O(n²) list membership check |
| 9 | Null reference | No bounds check on list index |
| 10 | — (correct) | Correct Fibonacci implementation |
| 11 | — (correct) | Correct dictionary merge |

---

## 3. Tasks

### Task 1 — Bug Detection *(Easy)*

**Objective:** Given a code snippet, respond `"yes"` (bug present) or `"no"` (bug absent).

**Grading function:**

```
R(a, g) = 1.0   if normalise(a) == ground_truth
          0.1   if a is non-empty but wrong
          0.0   if a is empty
```

**Episode horizon:** 3 steps. **Target agent accuracy:** ~90%.

---

### Task 2 — Bug Classification *(Medium)*

**Objective:** Classify the bug type from a closed vocabulary:
`{syntax_error, logic_error, null_reference, security_flaw, performance_issue}`.

**Grading function:**

```
R(a, g) = 1.0   if a == ground_truth                   (exact)
          0.5   if a ∈ related_group(ground_truth)     (partial)
          0.1   if a ∈ valid_categories \ {g}          (wrong but valid)
          0.0   if a ∉ valid_categories                (unparseable)
```

where `related_groups = [{logic_error, null_reference}, {security_flaw, performance_issue}]`.

**Episode horizon:** 5 steps. **Target agent accuracy:** ~60%.

---

### Task 3 — Code Fix *(Hard)*

**Objective:** Return corrected Python code that resolves the identified bug.

**Grading function (additive, max 1.0):**

```
R(a, g) = 0.4 × compiles(a)
        + 0.4 × (tests_passed(a, g) / total_tests(g))
        + 0.2 × static_clean(a)
```

- `compiles(a)` — 1 if the submitted code parses as valid Python, else 0
- `tests_passed / total_tests` — fraction of ground-truth assertions that hold
- `static_clean(a)` — 1 if no dangerous patterns (`eval`, `exec`, unsafe imports) are introduced

**Episode horizon:** 8 steps. **Target agent accuracy:** ~30%.

---

## 4. Server API

The environment is exposed over HTTP via a FastAPI server.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/reset` | Begin new episode: `{"task_id": "bug_detection"}` |
| `POST` | `/step` | Submit action: `{"response": "yes", "task_id": "bug_detection"}` |
| `GET` | `/state` | Full internal state snapshot |
| `GET` | `/tasks` | Task catalogue with action schemas |
| `POST` | `/grader` | Direct grader invocation for offline evaluation |
| `GET` | `/baseline` | Trigger baseline inference; returns per-task scores |

---

## 5. Baseline

A standalone inference script (`baseline.py`) evaluates GPT-4o-mini on all three tasks. When `OPENAI_API_KEY` is not set the script falls back to pre-computed mock responses, enabling offline verification of the grading pipeline.

### 5.1 Baseline Scores (Mock / Oracle)

| Task | Snippets | Average Score |
|------|----------|--------------|
| Bug Detection | 12 | 1.000 |
| Bug Classification | 10 | 1.000 |
| Code Fix | 10 | 0.960 |
| **Overall** | | **0.987** |

---

## 6. Setup and Deployment

### 6.1 Local

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### 6.2 Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

### 6.3 Hugging Face Spaces

Push this repository to a Hugging Face Space configured with the **Docker** SDK. The server listens on port 7860, which Spaces exposes automatically.

### 6.4 Baseline (with LLM)

```bash
export OPENAI_API_KEY=sk-...
python baseline.py
```

---

## 7. Example Interaction

```bash
# 1. Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "bug_detection"}'

# 2. Submit a response
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"response": "yes", "task_id": "bug_detection"}'

# 3. Inspect environment state
curl http://localhost:7860/state

# 4. Invoke a grader directly
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{
        "action": {"response": "logic_error", "task_id": "bug_classification"},
        "ground_truth": {"bug_type": "logic_error"}
      }'
```

---

## 8. Project Structure

```
.
├── openenv.yaml          # OpenEnv metadata manifest
├── models.py             # Pydantic v2 data models
├── environment.py        # CodeReviewEnv (reset / step / state)
├── server.py             # FastAPI HTTP server
├── baseline.py           # GPT-4o-mini baseline script
├── Dockerfile            # Container definition (port 7860)
├── requirements.txt      # Python dependencies
├── tasks/
│   ├── task1_easy.py     # Bug detection task module
│   ├── task2_medium.py   # Bug classification task module
│   └── task3_hard.py     # Code fix task module
└── graders/
    ├── grader_detection.py      # Grader: binary detection
    ├── grader_classification.py # Grader: multi-class classification
    └── grader_fix.py            # Grader: code repair (multi-component)
```

---

## 9. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥ 0.104 | HTTP server framework |
| `uvicorn` | ≥ 0.24 | ASGI server |
| `pydantic` | ≥ 2.5 | Data validation |
| `openai` | ≥ 1.6 | LLM baseline inference |
| `pyyaml` | ≥ 6.0 | YAML manifest parsing |
| `httpx` | ≥ 0.25 | Async HTTP client |

---

## 10. References

[1] OpenEnv Framework. *Building Real-World AI Agent Training Environments*. Meta × Scaler Hackathon, 2026.

[2] Chen, M. et al. *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374.

[3] Austin, J. et al. *Program Synthesis with Large Language Models*. arXiv:2108.07732.
