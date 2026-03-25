"""Grader for Task 2 — Bug Classification (medium).

Scoring:
    1.0  — exact match with the ground-truth bug type
    0.5  — related category (e.g., null_reference vs logic_error — both
           are correctness issues)
    0.1  — agent responded but answer is unrelated
    0.0  — empty response
"""

from __future__ import annotations

from typing import Any

from models import Action

# Groups of related bug categories. If the agent picks a category in the
# same group as the ground truth, they earn partial credit.
_RELATED_GROUPS: list[set[str]] = [
    {"logic_error", "null_reference"},       # both are correctness bugs
    {"security_flaw", "performance_issue"},   # both are non-functional concerns
    {"syntax_error"},                          # standalone
]

VALID_CATEGORIES = {
    "syntax_error",
    "logic_error",
    "null_reference",
    "security_flaw",
    "performance_issue",
}


class ClassificationGrader:
    """Grade the agent's bug-type classification."""

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        response = action.response.strip().lower().replace(" ", "_").replace("-", "_")
        expected: str = ground_truth.get("bug_type", "")

        if not response:
            return 0.0

        # Exact match
        if response == expected:
            return 1.0

        # Check if it's a valid category at all
        if response not in VALID_CATEGORIES:
            return 0.1

        # Related-category partial credit
        for group in _RELATED_GROUPS:
            if response in group and expected in group:
                return 0.5

        # Valid category but unrelated
        return 0.1
