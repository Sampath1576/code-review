"""Hard task — Multiple issue detection in Python code."""

from typing import Any

from code_review_env.models import Action, ActionType
from code_review_env.tasks.base_task import BaseTask


class HardTask(BaseTask):
    """Task Level 3: Detect multiple issues (performance, logic, security).

    Each snippet has 2-3 bugs. The agent submits one action per step.
    Each bug is scored up to 1.0:
      - Issue detected:     +0.40
      - Severity correct:   +0.35
      - Fix suggested:      +0.25
    Total per bug:          ~1.0 → final_score = avg across bugs

    Penalties:
      - False positive:     -0.20
      - Severity completely wrong: -0.10
    """

    @property
    def dataset_filename(self) -> str:
        return "hard.json"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def max_steps(self) -> int:
        return 10

    def grade(
        self, action: Action, snippet: dict[str, Any], found_bugs: list[int]
    ) -> tuple[float, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {
            "bug_detected": False,
            "severity_correct": False,
            "fix_suggested": False,
            "false_positive": False,
            "matched_bug_index": -1,
            "feedback": "",
        }

        if action.action_type == ActionType.APPROVE:
            info["feedback"] = "Code has multiple issues but agent approved it."
            return 0.001, info

        bugs = snippet.get("bugs", [])

        # Not identifying a bug — no reward
        if action.action_type not in (
            ActionType.IDENTIFY_BUG,
            ActionType.REQUEST_CHANGES,
            ActionType.SUGGEST_FIX,
            ActionType.CLASSIFY_SEVERITY,
        ):
            info["feedback"] = "Action type does not identify any issues."
            return 0.001, info

        remaining_bugs = [
            (idx, bug) for idx, bug in enumerate(bugs) if idx not in found_bugs
        ]

        if not remaining_bugs:
            info["false_positive"] = True
            info["feedback"] = "All bugs already found. No more issues to report."
            return 0.001, info

        # Find the best matching bug
        best_reward = -1.0
        best_info = info.copy()
        best_bug_idx = -1

        for idx, bug in remaining_bugs:
            step_reward = 0.0
            step_info = info.copy()

            # Score: does the agent's action match this bug?
            match_score = self._compute_match_score(action, bug)

            if match_score < 0.1:
                continue  # Not a match at all

            # 1. Issue detected
            step_reward += 0.40
            step_info["bug_detected"] = True

            # 2. Severity accuracy
            if action.severity is not None:
                expected_severity = bug.get("severity", "")
                if action.severity.value == expected_severity:
                    step_reward += 0.35
                    step_info["severity_correct"] = True
                else:
                    # Completely wrong severity (e.g., "minor" for "critical")
                    severity_distance = self._severity_distance(
                        action.severity.value, expected_severity
                    )
                    if severity_distance >= 2:
                        step_reward -= 0.10
                        step_info["severity_completely_wrong"] = True

            # 3. Fix suggestion
            if action.suggestion.strip():
                fix_keywords = bug.get("keywords", [])
                suggestion_lower = action.suggestion.lower()
                if fix_keywords and any(
                    kw.lower() in suggestion_lower for kw in fix_keywords
                ):
                    step_reward += 0.25
                    step_info["fix_suggested"] = True
                elif action.suggestion.strip():
                    step_reward += 0.10
                    step_info["fix_suggested"] = "partial"

            step_info["matched_bug_index"] = idx
            step_info["feedback"] = self._build_feedback(step_info, bug)

            if step_reward > best_reward:
                best_reward = step_reward
                best_info = step_info
                best_bug_idx = idx

        # No matching bug found — false positive
        if best_bug_idx == -1:
            best_info["false_positive"] = True
            best_info["feedback"] = "Reported issue does not match any remaining bug."
            return max(0.001, -0.20), best_info

        found_bugs.append(best_bug_idx)
        best_reward = max(0.001, min(0.999, best_reward))
        return best_reward, best_info

    def _compute_match_score(self, action: Action, bug: dict[str, Any]) -> float:
        """Compute how well the action matches a specific bug."""
        score = 0.0

        # Line number proximity
        expected_line = bug.get("line", 0)
        if expected_line > 0 and action.line_number > 0:
            distance = abs(action.line_number - expected_line)
            if distance == 0:
                score += 0.5
            elif distance <= 2:
                score += 0.3
            elif distance <= 4:
                score += 0.1

        # Bug type match
        if action.bug_type is not None and action.bug_type.value == bug.get("type", ""):
            score += 0.3

        # Keyword match in description
        keywords = bug.get("keywords", [])
        if keywords and action.description:
            desc_lower = action.description.lower()
            matched = sum(1 for kw in keywords if kw.lower() in desc_lower)
            score += 0.2 * min(1.0, matched / max(1, len(keywords)))

        return score

    @staticmethod
    def _severity_distance(actual: str, expected: str) -> int:
        """Distance between severity levels. 0=same, 1=adjacent, 2=opposite."""
        levels = ["minor", "major", "critical"]
        try:
            return abs(levels.index(actual) - levels.index(expected))
        except ValueError:
            return 2  # Unknown = maximum distance

    @staticmethod
    def _build_feedback(info: dict[str, Any], bug: dict[str, Any]) -> str:
        parts = []
        if info.get("bug_detected"):
            parts.append(f"✓ Detected {bug.get('type', 'unknown')} issue")
        if info.get("severity_correct"):
            parts.append("✓ Correct severity")
        if info.get("severity_completely_wrong"):
            parts.append("✗ Severity completely wrong (penalty applied)")
        if info.get("fix_suggested") is True:
            parts.append("✓ Good fix suggestion")
        elif info.get("fix_suggested") == "partial":
            parts.append("~ Partial fix")
        if not parts:
            parts.append(
                f"Expected: {bug.get('type', '?')} ({bug.get('severity', '?')}) "
                f"on line {bug.get('line', '?')}"
            )
        return " | ".join(parts)
