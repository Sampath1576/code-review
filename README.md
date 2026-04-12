# CodeReviewEnv

An OpenEnv reinforcement learning environment that trains AI agents to perform automated code review on Python code.

## 📊 Baseline Benchmark Results

| Difficulty | Episodes | Average Score | Perfect Scores |
|------------|----------|---------------|----------------|
| 🟢 Easy | 13 | **1.0000** | 13/13 (100%) |
| 🟡 Medium | 10 | **0.9850** | 9/10 (90%) |
| 🔴 Hard | 10 | **0.9483** | 7/10 (70%) |
| **Overall** | **33** | **~0.978** | **29/33** |

> Scores produced by the baseline agent in `inference.py` using LLM-augmented + rule-based detection.

## Overview

CodeReviewEnv presents Python code snippets to an AI agent, which must:
1. **Identify bugs** — find what's wrong in the code
2. **Classify severity** — rate issues as critical, major, or minor
3. **Suggest fixes** — propose how to fix each issue

The environment scores every action with a deterministic grading rubric and returns dense reward signals (0.0–1.0).

## Inference Agent

The baseline agent (`inference.py`) uses the **OpenAI API client** to run a model against the environment:

```python
# Environment variables read by inference.py:
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "...")  # API key
```

The agent first attempts LLM-based analysis, then supplements with rule-based pattern matching as a fallback for maximum detection coverage.

### Log Format

The agent emits the standardized log format:

```
[START] task=easy snippet=easy_001
[STEP] reward=1.0000 done=True step=1
[END] total_reward=1.0000 task=easy
```

## Difficulty Levels

| Level | Task | Bugs per Snippet | Max Steps | Focus |
|-------|------|-------------------|-----------|-------|
| 🟢 Easy | Syntax errors | 1 | 3 | Missing colons, brackets, parentheses, indentation |
| 🟡 Medium | Logic bugs | 1 | 5 | Off-by-one, wrong variables, edge cases |
| 🔴 Hard | Multiple issues | 2–3 | 10 | Security, performance, and logic combined |

## Action Space

```python
{
    "action_type": "identify_bug",      # identify_bug | classify_severity | suggest_fix | approve | request_changes
    "line_number": 3,                    # 1-indexed line number
    "bug_type": "syntax",               # syntax | logic | performance | security | style
    "severity": "critical",             # critical | major | minor
    "description": "Missing colon...",  # explanation of the issue
    "suggestion": "Add ':' at end..."   # proposed fix
}
```

## Observation Space

```python
{
    "code_snippet": "def foo(a, b)\n    ...",  # Python code to review
    "language": "python",
    "task_id": "easy_001",
    "step_count": 0,
    "context": "This function adds two numbers"
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode — accepts `{difficulty, snippet_id}` |
| `/step` | POST | Submit action — returns `{observation, reward, done, info}` |
| `/state` | GET | Current environment state |
| `/api/snippets/{difficulty}` | GET | List all snippets for a difficulty |
| `/api/run-episode` | POST | Run full agent episode with step-by-step trace |
| `/api/analyze-custom` | POST | Analyze user-submitted Python code |
| `/docs` | GET | Interactive Swagger API documentation |

## Quick Start

### Local Mode (no server)

```bash
pip install -r requirements.txt
python inference.py --difficulty easy
```

### Server Mode

```bash
# Terminal 1: Start the server
uvicorn server:app --host 0.0.0.0 --port 7860

# Terminal 2: Run the agent
python inference.py --server http://localhost:7860 --difficulty all
```

### Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your-api-key \
  code-review-env
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | Base URL for the OpenAI-compatible API |
| `MODEL_NAME` | `gpt-4o-mini` | Model name for inference |
| `HF_TOKEN` | — | API key / Hugging Face token |

## Reward Function

Rewards are dense and deterministic. Each component contributes partial credit:

### Easy Tasks
| Component | Points |
|-----------|--------|
| Bug detected | +0.40 |
| Exact line match | +0.30 |
| Line within ±2 | +0.20 |
| Correct bug type | +0.20 |
| Valid fix suggestion | +0.10 |

### Medium Tasks
| Component | Points |
|-----------|--------|
| Bug detected | +0.30 |
| Correct line | +0.20 |
| Good explanation | +0.30 |
| Correct fix | +0.20 |

### Hard Tasks (per bug)
| Component | Points |
|-----------|--------|
| Issue detected | +0.40 |
| Severity correct | +0.35 |
| Fix suggested | +0.25 |

### Penalties
| Condition | Penalty |
|-----------|---------|
| False positive | −0.20 |
| Severity completely wrong | −0.10 |
| Extra steps beyond optimal | −0.05/step |

## Dataset

- **33 code snippets** total (13 easy + 10 medium + 10 hard)
- All embedded as JSON files in `code_review_env/tasks/dataset/`
- No external database required

## Project Structure

```
code-review-env/
├── openenv.yaml              # OpenEnv manifest (tasks, env_vars, baseline)
├── Dockerfile                # Container config
├── requirements.txt          # Python dependencies (fastapi, pydantic, openai)
├── server.py                 # FastAPI server (9 endpoints)
├── inference.py              # LLM + rule-based agent
├── run_all.py                # Batch evaluation script
├── README.md                 # This file
├── static/
│   └── index.html            # Live dashboard UI
└── code_review_env/
    ├── __init__.py
    ├── env.py                # Main environment (step/reset/state)
    ├── models.py             # Pydantic models (Action, Observation, etc.)
    └── tasks/
        ├── __init__.py
        ├── base_task.py      # Abstract base class
        ├── task_easy.py      # Syntax error grading rubric
        ├── task_medium.py    # Logic bug grading rubric
        ├── task_hard.py      # Multi-issue grading rubric
        └── dataset/
            ├── easy.json     # 13 easy snippets
            ├── medium.json   # 10 medium snippets
            └── hard.json     # 10 hard snippets
```

## License

MIT
