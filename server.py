"""FastAPI server exposing the Scheduling Optimisation Environment as an HTTP API."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import SchedulingOptEnv
from graders.grader_classification import ConflictGrader
from graders.grader_detection import FeasibilityGrader
from graders.grader_fix import RepairGrader
from models import Action, Observation

app = FastAPI(
    title="Scheduling Optimisation Environment",
    description=(
        "OpenEnv-compatible environment for training AI agents on combinatorial "
        "scheduling optimisation problems."
    ),
    version="1.0.0",
)

# Single shared environment instance.
env = SchedulingOptEnv()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = "feasibility_check"


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class GradeRequest(BaseModel):
    action: Action
    ground_truth: dict[str, Any]


class GradeResponse(BaseModel):
    score: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health check for Hugging Face Spaces liveness probe."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None) -> Observation:
    """Reset the environment and start a new episode.

    Body (optional): {"task_id": "feasibility_check" | "conflict_classification" | "schedule_repair"}
    If no body is provided, defaults to "feasibility_check".
    """
    task_id = req.task_id if req else "feasibility_check"
    valid_tasks = {"feasibility_check", "conflict_classification", "schedule_repair"}
    if task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id. Choose from: {sorted(valid_tasks)}",
        )
    return env.reset(task_id=task_id)


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    """Submit an action and advance the environment by one step.

    Body: {"response": "<answer>", "task_id": "<task_id>"}
    """
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state() -> dict[str, Any]:
    """Return the full current environment state."""
    return env.state()


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    """List available tasks with their action schemas."""
    return [
        {
            "task_id": "feasibility_check",
            "name": "Feasibility Check",
            "difficulty": "easy",
            "max_steps": 3,
            "action_schema": {
                "response": "feasible | infeasible",
                "task_id": "feasibility_check",
            },
        },
        {
            "task_id": "conflict_classification",
            "name": "Conflict Classification",
            "difficulty": "medium",
            "max_steps": 5,
            "action_schema": {
                "response": (
                    "resource_overload | deadline_violation | precedence_violation | "
                    "availability_conflict | capacity_exceeded"
                ),
                "task_id": "conflict_classification",
            },
        },
        {
            "task_id": "schedule_repair",
            "name": "Schedule Repair",
            "difficulty": "hard",
            "max_steps": 8,
            "action_schema": {
                "response": '{"assignments": [{"job_id": "J1", "machine_id": "M1", "start_time": 0}, ...]}',
                "task_id": "schedule_repair",
            },
        },
    ]


@app.post("/grader", response_model=GradeResponse)
def grader(req: GradeRequest) -> GradeResponse:
    """Directly invoke a grader with an action and ground truth.

    Body: {"action": {"response": "...", "task_id": "..."}, "ground_truth": {...}}
    """
    task_id = req.action.task_id
    grader_map = {
        "feasibility_check": FeasibilityGrader(),
        "conflict_classification": ConflictGrader(),
        "schedule_repair": RepairGrader(),
    }
    g = grader_map.get(task_id)
    if g is None:
        raise HTTPException(
            status_code=400, detail=f"No grader for task_id={task_id}"
        )
    score = g.grade(req.action, req.ground_truth)
    return GradeResponse(score=max(0.0, min(1.0, score)))


@app.get("/baseline")
def baseline() -> dict[str, Any]:
    """Trigger the baseline inference agent and return per-task scores.

    Falls back to mock oracle responses when OPENAI_API_KEY is not set,
    so this endpoint always returns a valid result.
    """
    try:
        from baseline import run_baseline
        return run_baseline()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Baseline run failed: {exc}",
        ) from exc
