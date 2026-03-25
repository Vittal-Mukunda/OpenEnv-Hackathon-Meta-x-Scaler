"""Grader for Task 3 — Code Fix (hard).

Scoring breakdown (max 1.0):
    0.4  — the fixed code compiles (exec without SyntaxError/Exception)
    0.4  — the fixed code passes all ground-truth unit tests
    0.2  — no new obvious issues detected (basic static checks)

Partial-progress signal:
    0.2  — base reward if the submission is at least syntactically valid
           Python, even before running tests.
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any

from models import Action


class FixGrader:
    """Grade the agent's proposed code fix."""

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        code = action.response.strip()
        tests: list[str] = ground_truth.get("tests", [])

        if not code:
            return 0.0

        score = 0.0

        # ------------------------------------------------------------------
        # Component 1: Does the code compile? (0.4 pts)
        # ------------------------------------------------------------------
        compiles = self._compiles(code)
        if compiles:
            score += 0.4

        # ------------------------------------------------------------------
        # Component 2: Does it pass the ground-truth tests? (0.4 pts)
        # Each test is worth an equal share of 0.4.
        # ------------------------------------------------------------------
        if compiles and tests:
            passed = self._run_tests(code, tests)
            score += 0.4 * (passed / len(tests))

        # ------------------------------------------------------------------
        # Component 3: No new issues — basic static checks (0.2 pts)
        # We verify:
        #   - no bare `exec`/`eval` calls introduced
        #   - no `import os` / `import subprocess` (crude safety net)
        #   - the AST is structurally valid
        # ------------------------------------------------------------------
        if compiles:
            clean = self._static_check(code)
            if clean:
                score += 0.2

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, round(score, 4)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compiles(code: str) -> bool:
        """Return True if `code` parses as valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _run_tests(code: str, tests: list[str]) -> int:
        """Execute each test assertion against the submitted code.

        Returns the number of tests that passed.
        """
        passed = 0
        for test in tests:
            try:
                combined = textwrap.dedent(code) + "\n" + test
                exec(compile(combined, "<agent-fix>", "exec"), {})  # noqa: S102
                passed += 1
            except Exception:
                continue
        return passed

    @staticmethod
    def _static_check(code: str) -> bool:
        """Basic static analysis — reject obviously dangerous patterns."""
        lowered = code.lower()
        dangerous = ["eval(", "exec(", "import os", "import subprocess", "__import__"]
        for pattern in dangerous:
            if pattern in lowered:
                return False
        return True
