"""Pydantic models for CodeReviewEnv — defines the complete type-safe contract."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """Actions the agent can take when reviewing code."""
    IDENTIFY_BUG = "identify_bug"
    CLASSIFY_SEVERITY = "classify_severity"
    SUGGEST_FIX = "suggest_fix"
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"


class BugType(str, Enum):
    """Categories of bugs the agent can identify."""
    SYNTAX = "syntax"
    LOGIC = "logic"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STYLE = "style"


class Severity(str, Enum):
    """Severity levels for identified issues."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


# ── Action (what the agent submits) ──────────────────────────────────────────

class Action(BaseModel):
    """An action submitted by the agent during code review."""
    action_type: ActionType = Field(
        ...,
        description="The type of review action being taken"
    )
    line_number: int = Field(
        default=0,
        ge=0,
        description="The line number where the issue is located (1-indexed, 0 = not applicable)"
    )
    bug_type: Optional[BugType] = Field(
        default=None,
        description="The category of bug identified"
    )
    severity: Optional[Severity] = Field(
        default=None,
        description="The severity of the identified issue"
    )
    description: str = Field(
        default="",
        description="Explanation of the issue found"
    )
    suggestion: str = Field(
        default="",
        description="Suggested fix for the issue"
    )


# ── Observation (what the agent sees) ────────────────────────────────────────

class Observation(BaseModel):
    """The observation presented to the agent at each step."""
    code_snippet: str = Field(
        ...,
        description="The Python code to review"
    )
    language: str = Field(
        default="python",
        description="Programming language of the snippet"
    )
    task_id: str = Field(
        ...,
        description="Unique identifier for this task (e.g. easy_001)"
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken so far in this episode"
    )
    context: str = Field(
        default="",
        description="Description of what the code is supposed to do"
    )


# ── Step Result ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """The result returned after the agent takes a step."""
    observation: Observation = Field(
        ...,
        description="The current observation after taking the action"
    )
    reward: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reward for this step (0.0 to 1.0)"
    )
    done: bool = Field(
        ...,
        description="Whether the episode has ended"
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic information"
    )


# ── Environment State ───────────────────────────────────────────────────────

class EnvState(BaseModel):
    """Snapshot of the current environment state."""
    current_task_id: str = Field(
        ...,
        description="ID of the current task being reviewed"
    )
    step_count: int = Field(
        ...,
        ge=0,
        description="Number of steps taken in the current episode"
    )
    total_reward: float = Field(
        ...,
        description="Accumulated reward for the current episode"
    )
    done: bool = Field(
        ...,
        description="Whether the current episode is finished"
    )
    max_steps: int = Field(
        ...,
        gt=0,
        description="Maximum steps allowed for this episode"
    )
    difficulty: str = Field(
        ...,
        description="Current difficulty level (easy, medium, hard)"
    )


# ── Request Models for API ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""
    difficulty: str = Field(
        default="easy",
        description="Difficulty level: easy, medium, or hard"
    )
    snippet_id: Optional[str] = Field(
        default=None,
        description="Specific snippet ID to load (random if not provided)"
    )
