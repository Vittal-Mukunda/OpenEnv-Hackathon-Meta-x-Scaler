"""Inference script for the Scheduling Optimisation Environment.

Emits exactly three line types per episode:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Required environment variables:
    API_BASE_URL  — Base URL for the OpenAI-compatible LiteLLM proxy endpoint
    API_KEY       — API key injected by the contest validator
    MODEL_NAME    — Model identifier to use for inference

Usage:
    API_BASE_URL=<proxy_url> API_KEY=<key> MODEL_NAME=gpt-4o-mini python inference.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

from environment import INSTANCE_BANK, SchedulingOptEnv
from models import Action

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = os.environ["API_KEY"]
BENCHMARK: str = "scheduling-opt-env"
SUCCESS_THRESHOLD: float = 0.95

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Structured log helpers (exact required format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitise action: collapse newlines and truncate to keep lines readable
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _llm(system: str, user: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return ""


# ---------------------------------------------------------------------------
# Per-task agent prompts
# ---------------------------------------------------------------------------


def _agent_feasibility(instance_str: str, instance_idx: int) -> str:
    return _llm(
        "You are a scheduling expert. Determine if the proposed schedule satisfies "
        "all constraints. Reply with ONLY 'feasible' or 'infeasible'. No extra text.",
        instance_str,
    )


def _agent_classification(instance_str: str, instance_idx: int) -> str:
    return _llm(
        "You are a scheduling expert. Identify the single constraint violation type. "
        "Reply with ONLY one of: resource_overload, deadline_violation, "
        "precedence_violation, availability_conflict, capacity_exceeded. No extra text.",
        instance_str,
    )


def _agent_repair(instance_str: str, instance_idx: int) -> str:
    return _llm(
        'You are a scheduling expert. Repair the infeasible schedule. Return ONLY a '
        'valid JSON object: {"assignments": [{"job_id": "...", "machine_id": "...", '
        '"start_time": <int>}, ...]}. No markdown, no explanation.',
        instance_str,
    )


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "feasibility_check":      {"max_steps": 3,  "agent": _agent_feasibility},
    "conflict_classification":{"max_steps": 5,  "agent": _agent_classification},
    "schedule_repair":        {"max_steps": 8,  "agent": _agent_repair},
}


def run_episode(
    env: SchedulingOptEnv,
    task_id: str,
    instance_idx: int,
    instance_entry: dict,
) -> None:
    """Run one episode and emit [START] / [STEP]s / [END]."""
    cfg = TASK_CONFIG[task_id]
    max_steps: int = cfg["max_steps"]
    agent_fn = cfg["agent"]
    instance_str = json.dumps(instance_entry["instance"], indent=2)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset(task_id=task_id)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        for step in range(1, max_steps + 1):
            response = agent_fn(instance_str, instance_idx)
            action = Action(response=response, task_id=task_id)

            obs, reward, done, info = env.step(action)

            error = info.get("grading_breakdown", {}).get("feedback") if reward < SUCCESS_THRESHOLD else None
            # Only surface error string for failed/partial steps
            if reward >= SUCCESS_THRESHOLD:
                error = None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=response, reward=reward, done=done, error=error)

            if done:
                break

        final_reward = rewards[-1] if rewards else 0.0
        score = min(max(final_reward, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)
        if not rewards:
            rewards = [0.0]
        score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main — run all 32 episodes across 3 tasks
# ---------------------------------------------------------------------------


def main() -> None:
    env = SchedulingOptEnv()

    # Task 1: Feasibility Check — all 12 instances
    for i, entry in enumerate(INSTANCE_BANK):
        run_episode(env, "feasibility_check", i, entry)

    # Task 2: Conflict Classification — 10 infeasible instances only
    for i, entry in enumerate(INSTANCE_BANK):
        if not entry["is_feasible"]:
            run_episode(env, "conflict_classification", i, entry)

    # Task 3: Schedule Repair — 10 infeasible instances only
    for i, entry in enumerate(INSTANCE_BANK):
        if not entry["is_feasible"]:
            run_episode(env, "schedule_repair", i, entry)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
