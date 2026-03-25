"""Task 1 — Bug Detection (Easy).

The agent sees a Python code snippet and must respond with "yes" or "no"
to indicate whether the code contains a bug.

Grading: exact match — 1.0 if correct, 0.1 if wrong, 0.0 if empty.
Max steps per episode: 3.
Expected agent accuracy: ~90%.
"""

from __future__ import annotations

from typing import Any

from environment import SNIPPET_BANK, CodeReviewEnv
from graders.grader_detection import DetectionGrader
from models import Action

TASK_ID = "bug_detection"
MAX_STEPS = 3
DIFFICULTY = "easy"


def run_episode(env: CodeReviewEnv, agent_fn: Any) -> dict[str, Any]:
    """Run a single bug-detection episode.

    Args:
        env: An initialized CodeReviewEnv instance.
        agent_fn: A callable that receives an Observation and returns a
                  response string ("yes" or "no").

    Returns:
        Episode summary dict with total reward and step count.
    """
    obs = env.reset(task_id=TASK_ID)
    total_reward = 0.0
    steps = 0

    for _ in range(MAX_STEPS):
        response = agent_fn(obs)
        action = Action(response=response, task_id=TASK_ID)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    return {
        "task": TASK_ID,
        "difficulty": DIFFICULTY,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "info": info,
    }


def get_all_snippets_with_answers() -> list[dict[str, Any]]:
    """Return snippet bank entries relevant to bug detection."""
    return [
        {"code": s["code"], "has_bug": s["has_bug"], "description": s["description"]}
        for s in SNIPPET_BANK
    ]
