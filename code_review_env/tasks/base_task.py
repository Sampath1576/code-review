"""Abstract base class for all code review tasks."""

from abc import ABC, abstractmethod
import json
import os
from typing import Any

from code_review_env.models import Action, Observation


class BaseTask(ABC):
    """Base class for code review task definitions.

    Each task difficulty level subclasses this and implements:
      - get_snippets() to load the dataset
      - grade() to score the agent's action
    """

    def __init__(self) -> None:
        self._snippets: list[dict[str, Any]] = []
        self._load_snippets()

    def _load_snippets(self) -> None:
        """Load snippets from the JSON dataset file."""
        dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
        filepath = os.path.join(dataset_dir, self.dataset_filename)
        with open(filepath, "r", encoding="utf-8") as f:
            self._snippets = json.load(f)

    @property
    @abstractmethod
    def dataset_filename(self) -> str:
        """Name of the JSON dataset file (e.g. 'easy.json')."""
        ...

    @property
    @abstractmethod
    def difficulty(self) -> str:
        """Difficulty level identifier."""
        ...

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Maximum number of steps allowed per episode."""
        ...

    def get_snippets(self) -> list[dict[str, Any]]:
        """Return all available code snippets for this task level."""
        return self._snippets

    def get_snippet_by_id(self, snippet_id: str) -> dict[str, Any] | None:
        """Look up a specific snippet by its ID."""
        for snippet in self._snippets:
            if snippet["id"] == snippet_id:
                return snippet
        return None

    def get_observation(self, snippet: dict[str, Any], step_count: int) -> Observation:
        """Build an Observation from a snippet and the current step count."""
        return Observation(
            code_snippet=snippet["code"],
            language="python",
            task_id=snippet["id"],
            step_count=step_count,
            context=snippet.get("context", ""),
        )

    @abstractmethod
    def grade(
        self, action: Action, snippet: dict[str, Any], found_bugs: list[int]
    ) -> tuple[float, dict[str, Any]]:
        """Grade the agent's action against the expected bugs.

        Args:
            action: The action submitted by the agent.
            snippet: The current code snippet data (including expected bugs).
            found_bugs: Indices of bugs already found in previous steps.

        Returns:
            A tuple of (reward, info_dict) where reward is in [0.0, 1.0]
            and info_dict contains diagnostic feedback.
        """
        ...
