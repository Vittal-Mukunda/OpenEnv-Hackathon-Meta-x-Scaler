"""FastAPI server exposing the Code Review Environment as an HTTP API."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import CodeReviewEnv
from graders.grader_classification import ClassificationGrader
from graders.grader_detection import DetectionGrader
from graders.grader_fix import FixGrader
from models import Action, Observation

app = FastAPI(
    title="Code Review Environment",
    description="OpenEnv-compatible environment for AI-powered code review training.",
    version="1.0.0",
)

# Single shared environment instance.
env = CodeReviewEnv()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "bug_detection"


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
    """Health check for HF Spaces ping."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    """Reset the environment and start a new episode."""
    valid_tasks = {"bug_detection", "bug_classification", "code_fix"}
    if req.task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id. Choose from: {sorted(valid_tasks)}",
        )
    return env.reset(task_id=req.task_id)


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    """Submit an action and advance the environment by one step."""
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
            "task_id": "bug_detection",
            "name": "Bug Detection",
            "difficulty": "easy",
            "max_steps": 3,
            "action_schema": {"response": "yes | no", "task_id": "bug_detection"},
        },
        {
            "task_id": "bug_classification",
            "name": "Bug Classification",
            "difficulty": "medium",
            "max_steps": 5,
            "action_schema": {
                "response": "syntax_error | logic_error | null_reference | security_flaw | performance_issue",
                "task_id": "bug_classification",
            },
        },
        {
            "task_id": "code_fix",
            "name": "Code Fix",
            "difficulty": "hard",
            "max_steps": 8,
            "action_schema": {"response": "<corrected python code>", "task_id": "code_fix"},
        },
    ]


@app.post("/grader", response_model=GradeResponse)
def grader(req: GradeRequest) -> GradeResponse:
    """Directly invoke a grader with an action and ground truth."""
    task_id = req.action.task_id
    grader_map = {
        "bug_detection": DetectionGrader(),
        "bug_classification": ClassificationGrader(),
        "code_fix": FixGrader(),
    }
    g = grader_map.get(task_id)
    if g is None:
        raise HTTPException(status_code=400, detail=f"No grader for task_id={task_id}")
    score = g.grade(req.action, req.ground_truth)
    return GradeResponse(score=max(0.0, min(1.0, score)))


@app.get("/baseline")
def baseline() -> dict[str, Any]:
    """Run the baseline agent and return scores for all 3 tasks."""
    from baseline import run_baseline
    return run_baseline()
