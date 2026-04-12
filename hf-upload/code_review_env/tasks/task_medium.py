"""Medium task — Logic bug detection in Python code."""

from typing import Any

from code_review_env.models import Action, ActionType, BugType
from code_review_env.tasks.base_task import BaseTask


class MediumTask(BaseTask):
    """Task Level 2: Detect logic bugs in Python snippets.

    Grading rubric (max 1.0):
      - Bug detected:                   +0.30
      - Correct line identified:         +0.20
      - Explanation mentions key concept: +0.30
      - Fix is correct:                  +0.20
      - False positive penalty:          -0.20
    """

    @property
    def dataset_filename(self) -> str:
        return "medium.json"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def max_steps(self) -> int:
        return 5

    def grade(
        self, action: Action, snippet: dict[str, Any], found_bugs: list[int]
    ) -> tuple[float, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {
            "bug_detected": False,
            "line_correct": False,
            "explanation_quality": False,
            "fix_correct": False,
            "false_positive": False,
            "feedback": "",
        }

        if action.action_type == ActionType.APPROVE:
            info["feedback"] = "Code has a logic bug but agent approved it."
            return 0.001, info

        bugs = snippet.get("bugs", [])
        if not bugs:
            if action.action_type in (
                ActionType.IDENTIFY_BUG,
                ActionType.REQUEST_CHANGES,
                ActionType.SUGGEST_FIX,
            ):
                info["false_positive"] = True
                info["feedback"] = "No bugs exist but agent reported one."
                return 0.001, info
            return 0.001, info

        best_reward = 0.0
        best_info = info.copy()
        best_bug_idx = -1

        for idx, bug in enumerate(bugs):
            if idx in found_bugs:
                continue

            step_reward = 0.0
            step_info = info.copy()

            # 1. Bug detected
            if action.action_type in (
                ActionType.IDENTIFY_BUG,
                ActionType.REQUEST_CHANGES,
                ActionType.SUGGEST_FIX,
            ):
                step_reward += 0.30
                step_info["bug_detected"] = True

            # 2. Line accuracy
            expected_line = bug.get("line", 0)
            if action.line_number == expected_line:
                step_reward += 0.20
                step_info["line_correct"] = True
            elif abs(action.line_number - expected_line) <= 2:
                step_reward += 0.10
                step_info["line_correct"] = "approximate"

            # 3. Explanation quality — check if description mentions key concepts
            explanation_keywords = bug.get("keywords", [])
            description_lower = action.description.lower()
            if explanation_keywords:
                matched = sum(
                    1 for kw in explanation_keywords if kw.lower() in description_lower
                )
                if matched >= 2:
                    step_reward += 0.30
                    step_info["explanation_quality"] = True
                elif matched == 1:
                    step_reward += 0.15
                    step_info["explanation_quality"] = "partial"

            # 4. Fix correctness
            fix_keywords = bug.get("fix_keywords", bug.get("keywords", []))
            suggestion_lower = action.suggestion.lower()
            if suggestion_lower.strip():
                fix_matched = sum(
                    1 for kw in fix_keywords if kw.lower() in suggestion_lower
                )
                if fix_matched >= 1:
                    step_reward += 0.20
                    step_info["fix_correct"] = True
                elif suggestion_lower.strip():
                    step_reward += 0.05
                    step_info["fix_correct"] = "partial"

            step_info["feedback"] = self._build_feedback(step_info, bug)

            if step_reward > best_reward:
                best_reward = step_reward
                best_info = step_info
                best_bug_idx = idx

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
            parts.append("~ Approximate line match")
        if info.get("explanation_quality") is True:
            parts.append("✓ Good explanation")
        elif info.get("explanation_quality") == "partial":
            parts.append("~ Partial explanation")
        if info.get("fix_correct") is True:
            parts.append("✓ Correct fix")
        elif info.get("fix_correct") == "partial":
            parts.append("~ Partial fix")
        if not parts:
            parts.append(
                f"Expected: {bug.get('type', 'unknown')} bug on line {bug.get('line', '?')}"
            )
        return " | ".join(parts)
