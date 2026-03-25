"""Grader for Task 1 — Bug Detection (easy).

Scoring:
    1.0  — exact match with ground truth (yes/no)
    0.1  — agent gave a response but it was wrong (partial signal)
    0.0  — empty or unparseable response
"""

from __future__ import annotations

from typing import Any

from models import Action


class DetectionGrader:
    """Grade whether the agent correctly identified if a bug exists."""

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        response = action.response.strip().lower()
        has_bug: bool = ground_truth.get("has_bug", False)
        expected = "yes" if has_bug else "no"

        if not response:
            return 0.0

        # Normalize common variants
        positive = {"yes", "true", "1", "bug", "buggy"}
        negative = {"no", "false", "0", "clean", "correct", "no bug"}

        if response in positive:
            answer = "yes"
        elif response in negative:
            answer = "no"
        else:
            # Partial credit for trying — the agent responded but we
            # couldn't parse it cleanly.
            return 0.1

        # Exact match → full score; wrong answer → small partial signal
        return 1.0 if answer == expected else 0.1
