"""Easy task — Syntax error detection in Python code."""

from typing import Any

from code_review_env.models import Action, ActionType
from code_review_env.tasks.base_task import BaseTask


class EasyTask(BaseTask):
    """Task Level 1: Detect single syntax errors in Python snippets.

    Grading rubric (max 1.0):
      - Bug detected correctly:        +0.40
      - Line number exact match:        +0.30
      - Line number within ±2:          +0.20 (if not exact)
      - Bug type == 'syntax':           +0.20
      - Fix suggestion is valid:        +0.10
      - False positive penalty:         -0.20
    """

    @property
    def dataset_filename(self) -> str:
        return "easy.json"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def max_steps(self) -> int:
        return 3

    def grade(
        self, action: Action, snippet: dict[str, Any], found_bugs: list[int]
    ) -> tuple[float, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {
            "bug_detected": False,
            "line_correct": False,
            "type_correct": False,
            "fix_valid": False,
            "false_positive": False,
            "feedback": "",
        }

        # If the agent approves clean code — but this task always has bugs
        if action.action_type == ActionType.APPROVE:
            info["feedback"] = "Code has a syntax error but agent approved it."
            return 0.001, info

        bugs = snippet.get("bugs", [])
        if not bugs:
            # No bugs in snippet — any bug report is a false positive
            if action.action_type == ActionType.IDENTIFY_BUG:
                info["false_positive"] = True
                info["feedback"] = "No bugs exist but agent reported one."
                return 0.001, info
            return 0.001, info

        # Find the best matching bug for the agent's action
        best_reward = 0.0
        best_info = info.copy()
        best_bug_idx = -1

        for idx, bug in enumerate(bugs):
            if idx in found_bugs:
                continue  # Already found this one

            step_reward = 0.0
            step_info = info.copy()

            # 1. Bug detected (agent is reporting a bug)
            if action.action_type in (
                ActionType.IDENTIFY_BUG,
                ActionType.REQUEST_CHANGES,
                ActionType.SUGGEST_FIX,
            ):
                step_reward += 0.40
                step_info["bug_detected"] = True

            # 2. Line number accuracy
            expected_line = bug.get("line", 0)
            if action.line_number == expected_line:
                step_reward += 0.30
                step_info["line_correct"] = True
            elif abs(action.line_number - expected_line) <= 2:
                step_reward += 0.20
                step_info["line_correct"] = "approximate"

            # 3. Bug type classification
            if action.bug_type is not None and action.bug_type.value == bug.get("type", ""):
                step_reward += 0.20
                step_info["type_correct"] = True

            # 4. Fix suggestion quality
            if action.suggestion.strip():
                fix_keywords = bug.get("keywords", [])
                suggestion_lower = action.suggestion.lower()
                if fix_keywords and any(kw.lower() in suggestion_lower for kw in fix_keywords):
                    step_reward += 0.10
                    step_info["fix_valid"] = True
                elif action.suggestion.strip():
                    # Gave a suggestion but it doesn't match keywords — partial credit
                    step_reward += 0.05
                    step_info["fix_valid"] = "partial"

            step_info["feedback"] = self._build_feedback(step_info, bug)

            if step_reward > best_reward:
                best_reward = step_reward
                best_info = step_info
                best_bug_idx = idx

        # If agent reported a bug but none matched at all
        if best_bug_idx == -1 and action.action_type in (
            ActionType.IDENTIFY_BUG,
            ActionType.REQUEST_CHANGES,
            ActionType.SUGGEST_FIX,
        ):
            best_info["false_positive"] = True
            best_info["feedback"] = "Reported bug does not match any known issue."
            return 0.001, best_info

        if best_bug_idx >= 0:
            found_bugs.append(best_bug_idx)

        # Clamp reward
        best_reward = max(0.001, min(0.999, best_reward))
        return best_reward, best_info

    @staticmethod
    def _build_feedback(info: dict[str, Any], bug: dict[str, Any]) -> str:
        parts = []
        if info.get("bug_detected"):
            parts.append("✓ Bug detected")
        if info.get("line_correct") is True:
            parts.append("✓ Exact line match")
        elif info.get("line_correct") == "approximate":
            parts.append("~ Approximate line match (within ±2)")
        if info.get("type_correct"):
            parts.append("✓ Correct bug type")
        if info.get("fix_valid") is True:
            parts.append("✓ Valid fix suggestion")
        elif info.get("fix_valid") == "partial":
            parts.append("~ Partial fix credit")
        if not parts:
            parts.append(f"Expected: {bug.get('type', 'unknown')} error on line {bug.get('line', '?')}")
        return " | ".join(parts)
