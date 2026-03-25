"""Task 3 — Code Fix (Hard).

The agent sees buggy Python code and must provide corrected code that:
    (a) compiles without error                    — 0.4 pts
    (b) passes ground-truth unit tests            — 0.4 pts
    (c) introduces no new issues (static checks)  — 0.2 pts

Partial progress: syntactically valid submissions earn 0.2 base per step.
Max steps per episode: 8.
Expected agent accuracy: ~30%.
"""

from __future__ import annotations

from typing import Any

from environment import SNIPPET_BANK, CodeReviewEnv
from models import Action

TASK_ID = "code_fix"
MAX_STEPS = 8
DIFFICULTY = "hard"


def run_episode(env: CodeReviewEnv, agent_fn: Any) -> dict[str, Any]:
    """Run a single code-fix episode.

    Args:
        env: An initialized CodeReviewEnv instance.
        agent_fn: Callable receiving an Observation, returning fixed code string.

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


def get_fixable_snippets() -> list[dict[str, Any]]:
    """Return snippets that have bugs and known fixes."""
    return [
        {
            "code": s["code"],
            "fixed_code": s["fixed_code"],
            "tests": s["tests"],
            "description": s["description"],
        }
        for s in SNIPPET_BANK
        if s["has_bug"] and s["fixed_code"]
    ]
