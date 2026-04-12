"""CodeReviewEnv — Main environment implementing the OpenEnv step/reset/state interface."""

import random
from typing import Any, Optional

from code_review_env.models import (
    Action,
    ActionType,
    EnvState,
    Observation,
    StepResult,
)
from code_review_env.tasks.base_task import BaseTask
from code_review_env.tasks.task_easy import EasyTask
from code_review_env.tasks.task_medium import MediumTask
from code_review_env.tasks.task_hard import HardTask


class CodeReviewEnv:
    """An OpenEnv reinforcement learning environment for automated code review.

    The agent reads Python code snippets, identifies bugs, classifies their
    severity, and suggests fixes. The environment scores actions using fully
    deterministic grading rubrics.

    Interaction loop:
        1. reset(difficulty) → Observation
        2. step(action) → StepResult (observation, reward, done, info)
        3. state() → EnvState (current snapshot)
        4. Repeat until done
    """

    def __init__(self) -> None:
        # Task instances keyed by difficulty
        self._tasks: dict[str, BaseTask] = {
            "easy": EasyTask(),
            "medium": MediumTask(),
            "hard": HardTask(),
        }

        # Episode state
        self._current_task: Optional[BaseTask] = None
        self._current_snippet: Optional[dict[str, Any]] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = True
        self._max_steps: int = 3
        self._difficulty: str = "easy"
        self._found_bugs: list[int] = []
        self._episode_history: list[dict[str, Any]] = []

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        difficulty: str = "easy",
        snippet_id: Optional[str] = None,
    ) -> Observation:
        """Start a new episode.

        Args:
            difficulty: Task difficulty level — 'easy', 'medium', or 'hard'.
            snippet_id: Optional specific snippet ID. Random if not provided.

        Returns:
            The initial Observation for the agent.

        Raises:
            ValueError: If difficulty is invalid or snippet_id not found.
        """
        if difficulty not in self._tasks:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. Choose from: {list(self._tasks.keys())}"
            )

        task = self._tasks[difficulty]
        self._current_task = task
        self._difficulty = difficulty
        self._max_steps = task.max_steps

        # Select snippet
        if snippet_id is not None:
            snippet = task.get_snippet_by_id(snippet_id)
            if snippet is None:
                available = [s["id"] for s in task.get_snippets()]
                raise ValueError(
                    f"Snippet '{snippet_id}' not found. Available: {available}"
                )
        else:
            snippets = task.get_snippets()
            snippet = random.choice(snippets)

        self._current_snippet = snippet

        # Reset episode state
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._found_bugs = []
        self._episode_history = []

        return task.get_observation(snippet, self._step_count)

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """Submit a review action and receive feedback.

        Args:
            action: The agent's review action.

        Returns:
            StepResult with observation, reward, done flag, and info dict.

        Raises:
            RuntimeError: If the episode hasn't been started or is already done.
        """
        if self._current_task is None or self._current_snippet is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1

        # Grade the action
        step_reward, info = self._current_task.grade(
            action, self._current_snippet, self._found_bugs
        )

        # Apply step penalties
        total_bugs = len(self._current_snippet.get("bugs", []))
        optimal_steps = max(1, total_bugs)  # Minimum steps needed

        if self._step_count > optimal_steps:
            extra = self._step_count - optimal_steps
            penalty = 0.05 * extra
            step_reward = max(0.001, step_reward - penalty)
            info["step_penalty"] = penalty

        # Clamp reward
        step_reward = max(0.001, min(0.999, step_reward))
        self._total_reward += step_reward

        # Record history
        self._episode_history.append({
            "step": self._step_count,
            "action_type": action.action_type.value,
            "reward": step_reward,
            "info": info,
        })

        # Check termination conditions
        all_bugs_found = len(self._found_bugs) >= total_bugs
        max_steps_reached = self._step_count >= self._max_steps
        agent_approved = action.action_type == ActionType.APPROVE

        if all_bugs_found or max_steps_reached or agent_approved:
            self._done = True

        # Build info
        info["step"] = self._step_count
        info["total_reward"] = round(self._total_reward, 4)
        info["bugs_found"] = len(self._found_bugs)
        info["total_bugs"] = total_bugs
        info["steps_remaining"] = max(0, self._max_steps - self._step_count)

        if self._done:
            info["episode_complete"] = True
            raw_score = self._total_reward / max(1, total_bugs)
            final_score = max(0.0001, min(0.9999, raw_score))
            info["final_score"] = round(final_score, 4)

        observation = self._current_task.get_observation(
            self._current_snippet, self._step_count
        )

        return StepResult(
            observation=observation,
            reward=round(step_reward, 4),
            done=self._done,
            info=info,
        )

    # ── state ────────────────────────────────────────────────────────────────

    def state(self) -> EnvState:
        """Return the current environment state snapshot.

        Returns:
            EnvState with current task ID, step count, reward, etc.

        Raises:
            RuntimeError: If no episode is active.
        """
        if self._current_snippet is None:
            raise RuntimeError("No active episode. Call reset() first.")

        return EnvState(
            current_task_id=self._current_snippet["id"],
            step_count=self._step_count,
            total_reward=round(self._total_reward, 4),
            done=self._done,
            max_steps=self._max_steps,
            difficulty=self._difficulty,
        )
