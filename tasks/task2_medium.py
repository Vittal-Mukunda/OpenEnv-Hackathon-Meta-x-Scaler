"""Task 2 — Bug Classification (Medium).

The agent sees buggy Python code and must classify the bug type as one of:
    syntax_error, logic_error, null_reference, security_flaw, performance_issue

Grading:
    1.0  — exact match
    0.5  — related category
    0.0  — wrong / empty
Max steps per episode: 5.
Expected agent accuracy: ~60%.
"""

from __future__ import annotations

from typing import Any

from environment import SNIPPET_BANK, CodeReviewEnv
from models import Action

TASK_ID = "bug_classification"
MAX_STEPS = 5
DIFFICULTY = "medium"


def run_episode(env: CodeReviewEnv, agent_fn: Any) -> dict[str, Any]:
    """Run a single bug-classification episode.

    Args:
        env: An initialized CodeReviewEnv instance.
        agent_fn: Callable receiving an Observation, returning a bug-type string.

    Returns:
        Episode summary dict.
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


def get_buggy_snippets() -> list[dict[str, Any]]:
    """Return only snippets that have bugs (for classification task)."""
    return [
        {"code": s["code"], "bug_type": s["bug_type"], "description": s["description"]}
        for s in SNIPPET_BANK
        if s["has_bug"]
    ]
