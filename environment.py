"""Core Code Review Environment implementing the OpenEnv API contract."""

from __future__ import annotations

import copy
from typing import Any

from models import Action, Observation

# ---------------------------------------------------------------------------
# Code snippet bank — 12 diverse snippets with ground-truth metadata.
# Each entry carries:
#   code        – the Python source shown to the agent
#   has_bug     – bool, for Task 1 (bug detection)
#   bug_type    – str, for Task 2 (bug classification)
#   fixed_code  – str, for Task 3 (code fix)
#   tests       – list[str], mini-assert strings evaluated against the fix
#   description – human-readable summary of the issue
# ---------------------------------------------------------------------------

SNIPPET_BANK: list[dict[str, Any]] = [
    # 0 — Off-by-one error (logic_error)
    {
        "code": (
            "def average(nums):\n"
            "    total = 0\n"
            "    for i in range(1, len(nums)):\n"
            "        total += nums[i]\n"
            "    return total / len(nums)\n"
        ),
        "has_bug": True,
        "bug_type": "logic_error",
        "fixed_code": (
            "def average(nums):\n"
            "    total = 0\n"
            "    for i in range(len(nums)):\n"
            "        total += nums[i]\n"
            "    return total / len(nums)\n"
        ),
        "tests": [
            "assert average([2, 4, 6]) == 4.0",
            "assert average([10]) == 10.0",
        ],
        "description": "Off-by-one: loop starts at index 1, skipping the first element.",
    },
    # 1 — SQL injection (security_flaw)
    {
        "code": (
            "def get_user(conn, username):\n"
            '    query = f"SELECT * FROM users WHERE name = \'{username}\'"\n'
            "    return conn.execute(query).fetchone()\n"
        ),
        "has_bug": True,
        "bug_type": "security_flaw",
        "fixed_code": (
            "def get_user(conn, username):\n"
            '    query = "SELECT * FROM users WHERE name = ?"\n'
            "    return conn.execute(query, (username,)).fetchone()\n"
        ),
        "tests": [],
        "description": "SQL injection via f-string interpolation.",
    },
    # 2 — Unhandled exception / null reference (null_reference)
    {
        "code": (
            "def first_word(text):\n"
            "    words = text.split()\n"
            "    return words[0]\n"
        ),
        "has_bug": True,
        "bug_type": "null_reference",
        "fixed_code": (
            "def first_word(text):\n"
            "    if not text or not text.strip():\n"
            '        return ""\n'
            "    words = text.split()\n"
            "    return words[0]\n"
        ),
        "tests": [
            'assert first_word("hello world") == "hello"',
            'assert first_word("") == ""',
            'assert first_word("   ") == ""',
        ],
        "description": "IndexError when text is empty — words list is empty.",
    },
    # 3 — Infinite loop (logic_error)
    {
        "code": (
            "def countdown(n):\n"
            "    result = []\n"
            "    while n > 0:\n"
            "        result.append(n)\n"
            "        # forgot to decrement n\n"
            "    return result\n"
        ),
        "has_bug": True,
        "bug_type": "logic_error",
        "fixed_code": (
            "def countdown(n):\n"
            "    result = []\n"
            "    while n > 0:\n"
            "        result.append(n)\n"
            "        n -= 1\n"
            "    return result\n"
        ),
        "tests": [
            "assert countdown(3) == [3, 2, 1]",
            "assert countdown(0) == []",
        ],
        "description": "Infinite loop — n is never decremented.",
    },
    # 4 — Wrong variable usage (logic_error)
    {
        "code": (
            "def greet(first_name, last_name):\n"
            '    return f"Hello, {first_name} {first_name}!"\n'
        ),
        "has_bug": True,
        "bug_type": "logic_error",
        "fixed_code": (
            "def greet(first_name, last_name):\n"
            '    return f"Hello, {first_name} {last_name}!"\n'
        ),
        "tests": [
            'assert greet("Jane", "Doe") == "Hello, Jane Doe!"',
        ],
        "description": "Uses first_name twice instead of last_name.",
    },
    # 5 — Missing return statement (logic_error)
    {
        "code": (
            "def is_even(n):\n"
            "    if n % 2 == 0:\n"
            "        return True\n"
            "    # missing return False\n"
        ),
        "has_bug": True,
        "bug_type": "logic_error",
        "fixed_code": (
            "def is_even(n):\n"
            "    if n % 2 == 0:\n"
            "        return True\n"
            "    return False\n"
        ),
        "tests": [
            "assert is_even(4) is True",
            "assert is_even(3) is False",
        ],
        "description": "Returns None for odd numbers — missing else branch.",
    },
    # 6 — Type mismatch (logic_error)
    {
        "code": (
            "def add_tax(price, tax_rate):\n"
            '    return price + tax_rate + "%"\n'
        ),
        "has_bug": True,
        "bug_type": "logic_error",
        "fixed_code": (
            "def add_tax(price, tax_rate):\n"
            "    return price * (1 + tax_rate / 100)\n"
        ),
        "tests": [
            "assert abs(add_tax(100, 10) - 110.0) < 0.01",
        ],
        "description": "TypeError: concatenating number with string.",
    },
    # 7 — Insecure randomness (security_flaw)
    {
        "code": (
            "import random\n"
            "\n"
            "def generate_token(length=32):\n"
            '    chars = "abcdefghijklmnopqrstuvwxyz0123456789"\n'
            '    return "".join(random.choice(chars) for _ in range(length))\n'
        ),
        "has_bug": True,
        "bug_type": "security_flaw",
        "fixed_code": (
            "import secrets\n"
            "import string\n"
            "\n"
            "def generate_token(length=32):\n"
            "    chars = string.ascii_lowercase + string.digits\n"
            '    return "".join(secrets.choice(chars) for _ in range(length))\n'
        ),
        "tests": [
            "assert len(generate_token()) == 32",
            "assert len(generate_token(16)) == 16",
        ],
        "description": "Using random (Mersenne Twister) for security tokens instead of secrets.",
    },
    # 8 — Performance issue: O(n²) membership check (performance_issue)
    {
        "code": (
            "def unique_items(items):\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item not in result:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        "has_bug": True,
        "bug_type": "performance_issue",
        "fixed_code": (
            "def unique_items(items):\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item not in seen:\n"
            "            seen.add(item)\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        "tests": [
            "assert unique_items([1, 2, 2, 3, 1]) == [1, 2, 3]",
            "assert unique_items([]) == []",
        ],
        "description": "O(n²) due to `in` on a list; use a set for lookups.",
    },
    # 9 — Buffer/index overflow concept (logic_error)
    {
        "code": (
            "def safe_get(lst, index):\n"
            "    return lst[index]\n"
        ),
        "has_bug": True,
        "bug_type": "null_reference",
        "fixed_code": (
            "def safe_get(lst, index, default=None):\n"
            "    if 0 <= index < len(lst):\n"
            "        return lst[index]\n"
            "    return default\n"
        ),
        "tests": [
            "assert safe_get([10, 20], 0) == 10",
            "assert safe_get([10, 20], 5) is None",
            "assert safe_get([], 0) is None",
        ],
        "description": "No bounds checking — IndexError on out-of-range index.",
    },
    # 10 — Correct code (no bug)
    {
        "code": (
            "def fibonacci(n):\n"
            "    if n <= 0:\n"
            "        return []\n"
            "    if n == 1:\n"
            "        return [0]\n"
            "    seq = [0, 1]\n"
            "    while len(seq) < n:\n"
            "        seq.append(seq[-1] + seq[-2])\n"
            "    return seq\n"
        ),
        "has_bug": False,
        "bug_type": "none",
        "fixed_code": None,
        "tests": [
            "assert fibonacci(5) == [0, 1, 1, 2, 3]",
            "assert fibonacci(0) == []",
        ],
        "description": "Correct Fibonacci implementation — no bug.",
    },
    # 11 — Correct code (no bug)
    {
        "code": (
            "def merge_dicts(a, b):\n"
            "    result = {**a, **b}\n"
            "    return result\n"
        ),
        "has_bug": False,
        "bug_type": "none",
        "fixed_code": None,
        "tests": [
            'assert merge_dicts({"x": 1}, {"y": 2}) == {"x": 1, "y": 2}',
        ],
        "description": "Correct dictionary merge — no bug.",
    },
]


class CodeReviewEnv:
    """OpenEnv-compatible code review environment.

    API:
        reset(task_id)  → Observation
        step(action)    → (Observation, float, bool, dict)
        state()         → dict
    """

    def __init__(self) -> None:
        self._task_id: str = ""
        self._step: int = 0
        self._max_steps: int = 3
        self._snippet_index: int = 0
        self._done: bool = True
        self._history: list[dict[str, Any]] = []
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "bug_detection") -> Observation:
        """Start a new episode for the given task.

        Cycles through the snippet bank so repeated resets yield fresh
        snippets.
        """
        self._task_id = task_id
        self._step = 0
        self._done = False
        self._history = []
        self._cumulative_reward = 0.0

        # Determine max steps based on task difficulty
        step_limits = {
            "bug_detection": 3,
            "bug_classification": 5,
            "code_fix": 8,
        }
        self._max_steps = step_limits.get(task_id, 3)

        # Pick next snippet (round-robin)
        snippet = SNIPPET_BANK[self._snippet_index % len(SNIPPET_BANK)]
        self._snippet_index += 1

        context = self._context_for_task(task_id)

        return Observation(
            code_snippet=snippet["code"],
            task_id=task_id,
            context=context,
            step_number=self._step,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Process one agent action and return (obs, reward, done, info)."""
        if self._done:
            # Episode already finished — return terminal observation with 0 reward
            obs = Observation(
                code_snippet="",
                task_id=self._task_id,
                context="Episode is over. Call reset() to start a new one.",
                step_number=self._step,
            )
            return obs, 0.0, True, {"error": "episode_already_done"}

        self._step += 1

        # Resolve the current snippet (the one served at reset or last step)
        snippet = SNIPPET_BANK[(self._snippet_index - 1) % len(SNIPPET_BANK)]

        # Grade the action
        from graders.grader_detection import DetectionGrader
        from graders.grader_classification import ClassificationGrader
        from graders.grader_fix import FixGrader

        grader_map = {
            "bug_detection": DetectionGrader(),
            "bug_classification": ClassificationGrader(),
            "code_fix": FixGrader(),
        }
        grader = grader_map.get(self._task_id, DetectionGrader())
        reward = grader.grade(action, snippet)

        # Clamp reward to [0.0, 1.0]
        reward = max(0.0, min(1.0, reward))
        self._cumulative_reward += reward

        # Check termination
        done = self._step >= self._max_steps or reward >= 0.95
        self._done = done

        # Record history
        self._history.append({
            "step": self._step,
            "action": action.response,
            "reward": reward,
        })

        # Build next observation — either terminal or next snippet
        if done:
            obs = Observation(
                code_snippet="",
                task_id=self._task_id,
                context="Episode complete." if reward >= 0.95 else "Max steps reached.",
                step_number=self._step,
            )
        else:
            obs = Observation(
                code_snippet=snippet["code"],
                task_id=self._task_id,
                context=self._context_for_task(self._task_id),
                step_number=self._step,
            )

        info = {
            "cumulative_reward": round(self._cumulative_reward, 4),
            "steps_remaining": max(0, self._max_steps - self._step),
            "snippet_description": snippet["description"],
        }
        return obs, round(reward, 4), done, info

    def state(self) -> dict[str, Any]:
        """Return full current environment state."""
        snippet = SNIPPET_BANK[(self._snippet_index - 1) % len(SNIPPET_BANK)] if self._snippet_index > 0 else {}
        return {
            "task_id": self._task_id,
            "step": self._step,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "history": copy.deepcopy(self._history),
            "current_snippet": snippet.get("code", ""),
            "snippet_index": self._snippet_index,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _context_for_task(task_id: str) -> str:
        contexts = {
            "bug_detection": (
                "Review the code snippet. Respond with exactly 'yes' if there "
                "is a bug, or 'no' if the code is correct."
            ),
            "bug_classification": (
                "The code contains a bug. Classify it as one of: "
                "syntax_error, logic_error, null_reference, security_flaw, "
                "performance_issue."
            ),
            "code_fix": (
                "The code contains a bug. Provide the corrected Python code "
                "that fixes the issue. Return only the fixed code."
            ),
        }
        return contexts.get(task_id, "Review the code snippet.")

    @staticmethod
    def get_snippet_bank() -> list[dict[str, Any]]:
        """Expose the snippet bank for external use (e.g., baseline)."""
        return SNIPPET_BANK
