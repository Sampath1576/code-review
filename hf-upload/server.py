"""FastAPI server for CodeReviewEnv — exposes the OpenEnv HTTP interface + live dashboard."""

import sys
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Any

from code_review_env.env import CodeReviewEnv
from code_review_env.models import (
    Action,
    ActionType,
    EnvState,
    Observation,
    ResetRequest,
    StepResult,
)

# Add project root to path so we can import inference
sys.path.insert(0, os.path.dirname(__file__))
from inference import (
    analyze_code,
    build_action,
    detect_syntax_errors,
    detect_logic_errors,
    detect_security_issues,
    detect_performance_issues,
    detect_hard_logic,
)

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CodeReviewEnv",
    description=(
        "An OpenEnv reinforcement learning environment for training AI agents "
        "to perform automated code review on Python code. The agent identifies "
        "bugs, classifies severity, and suggests fixes across three difficulty levels."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance
env = CodeReviewEnv()

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the live dashboard."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Mount static files AFTER the root route
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Original Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "environment": "CodeReviewEnv", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()) -> Observation:
    """Start a new episode."""
    try:
        observation = env.reset(
            difficulty=request.difficulty,
            snippet_id=request.snippet_id,
        )
        return observation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """Submit a review action."""
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvState)
def get_state() -> EnvState:
    """Get current environment state."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Dashboard API Endpoints ──────────────────────────────────────────────────

@app.get("/api/snippets/{difficulty}")
def get_snippets(difficulty: str) -> dict:
    """Get all snippet IDs and contexts for a difficulty level."""
    if difficulty not in env._tasks:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}")

    snippets = env._tasks[difficulty].get_snippets()
    return {
        "difficulty": difficulty,
        "count": len(snippets),
        "snippets": [
            {"id": s["id"], "context": s.get("context", "")}
            for s in snippets
        ],
    }


class RunEpisodeRequest(BaseModel):
    """Request body for /api/run-episode."""
    difficulty: str = Field(default="easy")
    snippet_id: Optional[str] = Field(default=None)


@app.post("/api/run-episode")
def run_episode(request: RunEpisodeRequest) -> dict:
    """Run a full agent episode and return all steps with results.

    This endpoint resets the environment, runs the inference agent
    step-by-step, and returns the complete analysis trace.
    """
    difficulty = request.difficulty
    snippet_id = request.snippet_id

    if difficulty not in env._tasks:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}")

    # Reset environment
    try:
        obs = env.reset(difficulty=difficulty, snippet_id=snippet_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run inference agent
    found_issues = analyze_code(obs.code_snippet, difficulty, context=obs.context)
    steps: list[dict[str, Any]] = []

    if not found_issues:
        # Agent approves the code
        action = Action(action_type=ActionType.APPROVE, description="No issues found")
        result = env.step(action)
        steps.append({
            "step": 1,
            "action_type": "approve",
            "bug_type": None,
            "severity": None,
            "line_number": 0,
            "description": "No issues found — code looks clean.",
            "suggestion": "",
            "reward": result.reward,
            "feedback": result.info.get("feedback", ""),
        })
    else:
        for n, issue in enumerate(found_issues):
            if env._done:
                break
            action = Action(**build_action(issue))
            result = env.step(action)
            steps.append({
                "step": n + 1,
                "action_type": issue.get("type", "unknown"),
                "bug_type": issue.get("type", None),
                "severity": issue.get("severity", None),
                "line_number": issue.get("line", 0),
                "description": issue.get("description", ""),
                "suggestion": issue.get("suggestion", ""),
                "reward": result.reward,
                "feedback": result.info.get("feedback", ""),
            })
            if result.done:
                break

    # Get final state
    state = env.state()
    final_info = result.info if steps else {}
    final_score = final_info.get("final_score", state.total_reward)
    bugs_found = final_info.get("bugs_found", 0)
    total_bugs = final_info.get("total_bugs", 0)

    return {
        "task_id": obs.task_id,
        "difficulty": difficulty,
        "code": obs.code_snippet,
        "context": obs.context,
        "steps": steps,
        "final_score": round(final_score, 4),
        "total_reward": round(state.total_reward, 4),
        "bugs_found": bugs_found,
        "total_bugs": total_bugs,
        "episode_complete": True,
    }


# ── Custom Code Analysis ─────────────────────────────────────────────────────

class CustomCodeRequest(BaseModel):
    """Request body for /api/analyze-custom."""
    code: str = Field(..., description="Python code to analyze")


@app.post("/api/analyze-custom")
def analyze_custom_code(request: CustomCodeRequest) -> dict:
    """Analyze user-submitted Python code with all available detectors.

    Runs syntax, logic, security, performance, and hard-logic detectors
    and returns a comprehensive analysis.
    """
    code = request.code
    if not code.strip():
        return {"issues": [], "summary": "No code provided."}

    # Run ALL detectors to get comprehensive results
    all_issues: list[dict[str, Any]] = []
    all_issues.extend(detect_syntax_errors(code))
    all_issues.extend(detect_logic_errors(code))
    all_issues.extend(detect_security_issues(code))
    all_issues.extend(detect_performance_issues(code))
    all_issues.extend(detect_hard_logic(code))

    # Deduplicate: keep highest severity per line
    severity_rank = {"critical": 0, "major": 1, "minor": 2}
    by_line: dict[int, dict[str, Any]] = {}
    for issue in all_issues:
        ln = issue["line"]
        if ln not in by_line or severity_rank.get(issue["severity"], 9) < severity_rank.get(by_line[ln]["severity"], 9):
            by_line[ln] = issue

    deduped = sorted(by_line.values(), key=lambda x: severity_rank.get(x["severity"], 9))

    # Build steps in the same format as run-episode for frontend compatibility
    steps = []
    for i, issue in enumerate(deduped):
        steps.append({
            "step": i + 1,
            "action_type": issue.get("type", "unknown"),
            "bug_type": issue.get("type", None),
            "severity": issue.get("severity", None),
            "line_number": issue.get("line", 0),
            "description": issue.get("description", ""),
            "suggestion": issue.get("suggestion", ""),
            "reward": 1.0,  # No grading for custom code
            "feedback": "",
        })

    # Summary stats
    severity_counts = {"critical": 0, "major": 0, "minor": 0}
    type_counts: dict[str, int] = {}
    for issue in deduped:
        sev = issue.get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        t = issue.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    if not deduped:
        summary = "✅ No issues detected — the code looks clean!"
    else:
        parts = []
        if severity_counts["critical"]:
            parts.append(f"{severity_counts['critical']} critical")
        if severity_counts["major"]:
            parts.append(f"{severity_counts['major']} major")
        if severity_counts["minor"]:
            parts.append(f"{severity_counts['minor']} minor")
        summary = f"Found {len(deduped)} issue{'s' if len(deduped) != 1 else ''}: {', '.join(parts)}"

    return {
        "code": code,
        "steps": steps,
        "total_issues": len(deduped),
        "severity_counts": severity_counts,
        "type_counts": type_counts,
        "summary": summary,
    }

