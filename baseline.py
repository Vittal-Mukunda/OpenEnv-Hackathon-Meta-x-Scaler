"""Baseline inference script for the Code Review Environment.

Runs GPT-4o-mini (or falls back to mock responses) against all three tasks
and prints a score report.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from environment import CodeReviewEnv, SNIPPET_BANK
from graders.grader_classification import ClassificationGrader
from graders.grader_detection import DetectionGrader
from graders.grader_fix import FixGrader
from models import Action


def _get_openai_client():
    """Return an OpenAI client, or None if unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _llm_response(client, system_prompt: str, user_prompt: str) -> str:
    """Call the LLM and return the response text."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM error: {e}]")
        return ""


# ---------------------------------------------------------------------------
# Mock fallback responses (used when no API key is available)
# ---------------------------------------------------------------------------

_MOCK_DETECTION: dict[int, str] = {
    0: "yes", 1: "yes", 2: "yes", 3: "yes", 4: "yes",
    5: "yes", 6: "yes", 7: "yes", 8: "yes", 9: "yes",
    10: "no", 11: "no",
}

_MOCK_CLASSIFICATION: dict[int, str] = {
    0: "logic_error",
    1: "security_flaw",
    2: "null_reference",
    3: "logic_error",
    4: "logic_error",
    5: "logic_error",
    6: "logic_error",
    7: "security_flaw",
    8: "performance_issue",
    9: "null_reference",
}


def _mock_fix(snippet_idx: int) -> str:
    """Return the known fixed code for mock mode."""
    s = SNIPPET_BANK[snippet_idx]
    return s.get("fixed_code") or s["code"]


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def run_baseline() -> dict[str, Any]:
    """Execute the baseline across all tasks and return scores."""
    client = _get_openai_client()
    use_llm = client is not None
    mode = "GPT-4o-mini" if use_llm else "mock (no API key)"
    print(f"\n{'='*60}")
    print(f"  Code Review Env — Baseline ({mode})")
    print(f"{'='*60}\n")

    results: dict[str, Any] = {"mode": mode, "tasks": {}}

    # ----- Task 1: Bug Detection -----
    det_grader = DetectionGrader()
    det_scores: list[float] = []
    print("Task 1: Bug Detection (easy)")
    for i, snippet in enumerate(SNIPPET_BANK):
        if use_llm:
            resp = _llm_response(
                client,
                "You are a code reviewer. Answer ONLY 'yes' or 'no': does this code have a bug?",
                snippet["code"],
            )
        else:
            resp = _MOCK_DETECTION.get(i, "yes")
        action = Action(response=resp, task_id="bug_detection")
        score = det_grader.grade(action, snippet)
        det_scores.append(score)
        status = "CORRECT" if score >= 0.95 else "wrong"
        print(f"  Snippet {i:2d}: {status}  (score={score:.2f})  [{snippet['description'][:50]}]")

    avg_det = sum(det_scores) / len(det_scores) if det_scores else 0.0
    results["tasks"]["bug_detection"] = {
        "average_score": round(avg_det, 4),
        "num_snippets": len(det_scores),
        "scores": det_scores,
    }
    print(f"  >> Average: {avg_det:.3f}\n")

    # ----- Task 2: Bug Classification -----
    cls_grader = ClassificationGrader()
    cls_scores: list[float] = []
    buggy = [(i, s) for i, s in enumerate(SNIPPET_BANK) if s["has_bug"]]
    print("Task 2: Bug Classification (medium)")
    for i, snippet in buggy:
        if use_llm:
            resp = _llm_response(
                client,
                (
                    "You are a code reviewer. Classify the bug in this code as exactly one of: "
                    "syntax_error, logic_error, null_reference, security_flaw, performance_issue. "
                    "Reply with ONLY the category name."
                ),
                snippet["code"],
            )
        else:
            resp = _MOCK_CLASSIFICATION.get(i, "logic_error")
        action = Action(response=resp, task_id="bug_classification")
        score = cls_grader.grade(action, snippet)
        cls_scores.append(score)
        status = "EXACT" if score >= 0.95 else ("partial" if score >= 0.45 else "wrong")
        print(f"  Snippet {i:2d}: {status:7s} (score={score:.2f})  expected={snippet['bug_type']}")

    avg_cls = sum(cls_scores) / len(cls_scores) if cls_scores else 0.0
    results["tasks"]["bug_classification"] = {
        "average_score": round(avg_cls, 4),
        "num_snippets": len(cls_scores),
        "scores": cls_scores,
    }
    print(f"  >> Average: {avg_cls:.3f}\n")

    # ----- Task 3: Code Fix -----
    fix_grader = FixGrader()
    fix_scores: list[float] = []
    fixable = [(i, s) for i, s in enumerate(SNIPPET_BANK) if s["has_bug"] and s.get("fixed_code")]
    print("Task 3: Code Fix (hard)")
    for i, snippet in fixable:
        if use_llm:
            resp = _llm_response(
                client,
                (
                    "You are a code reviewer. Fix the bug in this Python code. "
                    "Return ONLY the corrected code, no explanations."
                ),
                snippet["code"],
            )
        else:
            resp = _mock_fix(i)
        action = Action(response=resp, task_id="code_fix")
        score = fix_grader.grade(action, snippet)
        fix_scores.append(score)
        print(f"  Snippet {i:2d}: score={score:.2f}  [{snippet['description'][:50]}]")

    avg_fix = sum(fix_scores) / len(fix_scores) if fix_scores else 0.0
    results["tasks"]["code_fix"] = {
        "average_score": round(avg_fix, 4),
        "num_snippets": len(fix_scores),
        "scores": fix_scores,
    }
    print(f"  >> Average: {avg_fix:.3f}\n")

    # ----- Summary -----
    overall = (avg_det + avg_cls + avg_fix) / 3
    results["overall_average"] = round(overall, 4)
    print(f"{'='*60}")
    print(f"  Overall Average: {overall:.3f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    try:
        run_baseline()
    except Exception as e:
        print(f"Baseline failed: {e}", file=sys.stderr)
        sys.exit(1)
