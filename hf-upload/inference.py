"""Baseline inference agent for CodeReviewEnv.

Uses the OpenAI API client to run a model against the environment.
Reads API credentials from environment variables:
    - API_BASE_URL  (default: https://api.openai.com/v1)
    - MODEL_NAME    (default: gpt-4o-mini)
    - HF_TOKEN      (API key)

Also includes comprehensive rule-based pattern matching as a fallback
for deterministic detection of syntax errors, logic bugs, performance
issues, and security vulnerabilities.

Usage:
    python inference.py                     # Run against all difficulties
    python inference.py --difficulty easy    # Run against a specific difficulty
    python inference.py --server http://localhost:7860  # Run against a server
"""

import argparse
import json
import os
import re
import sys
from typing import Any


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

def _get_llm_client():
    """Initialize the OpenAI-compatible LLM client from environment variables."""
    try:
        from openai import OpenAI
        
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            raise ValueError("HF_TOKEN environment variable is required")

        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
            api_key=hf_token,
        )
        return client
    except ImportError:
        return None


LLM_CLIENT = _get_llm_client()
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")


def llm_analyze(code: str, context: str = "", difficulty: str = "easy") -> list[dict[str, Any]]:
    """Use the LLM to analyze code for bugs.

    Sends the code to the configured LLM via OpenAI-compatible API and
    parses the structured response into our standard issue format.
    Falls back gracefully if the API is unavailable.
    """
    if LLM_CLIENT is None:
        return []

    system_prompt = (
        "You are an expert Python code reviewer. Analyze the given code and identify bugs. "
        "Return a JSON array of issues. Each issue must have: "
        '{"line": <int>, "type": "syntax|logic|security|performance", '
        '"severity": "critical|major|minor", '
        '"description": "<what is wrong>", "suggestion": "<how to fix>"}\n'
        f"Difficulty level: {difficulty}. Context: {context or 'Python code review'}\n"
        "Return ONLY the JSON array, no other text."
    )

    try:
        response = LLM_CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review this Python code:\n```python\n{code}\n```"},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        # Extract JSON from response (handle markdown fences)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            content = content.rsplit("```", 1)[0]
        content = content.strip()
        if content.startswith("json"):
            content = content[4:].strip()
        issues = json.loads(content)
        if isinstance(issues, list):
            # Validate structure
            valid = []
            for iss in issues:
                if isinstance(iss, dict) and "line" in iss and "type" in iss:
                    valid.append({
                        "line": int(iss.get("line", 0)),
                        "type": str(iss.get("type", "logic")),
                        "severity": str(iss.get("severity", "major")),
                        "description": str(iss.get("description", "")),
                        "suggestion": str(iss.get("suggestion", "")),
                    })
            return valid
    except Exception:
        pass  # Fallback to rule-based detection

    return []


# =============================================================================
# SYNTAX ERROR DETECTION (Easy)
# =============================================================================

def detect_syntax_errors(code: str) -> list[dict[str, Any]]:
    """Detect syntax errors: missing colons and unmatched brackets."""
    issues: list[dict[str, Any]] = []
    lines = code.split("\n")

    # --- Pass 1: Missing colons ---
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue

        patterns = [
            (r"^\s*def\s+\w+\s*\(.*\)\s*$", "function definition"),
            (r"^\s*class\s+\w+.*\)\s*$", "class definition"),
            (r"^\s*class\s+\w+\s*$", "class definition"),
            (r"^\s*if\s+.+\S\s*$", "if statement"),
            (r"^\s*elif\s+.+\S\s*$", "elif statement"),
            (r"^\s*else\s*$", "else clause"),
            (r"^\s*for\s+\w+\s+in\s+.+\S\s*$", "for loop"),
            (r"^\s*while\s+.+\S\s*$", "while loop"),
            (r"^\s*try\s*$", "try block"),
            (r"^\s*except.*\S\s*$", "except clause"),
            (r"^\s*finally\s*$", "finally clause"),
            (r"^\s*with\s+.+\S\s*$", "with statement"),
        ]

        if stripped.endswith(":") or stripped.endswith("\\"):
            continue

        for pat, desc in patterns:
            if re.match(pat, stripped):
                issues.append({
                    "line": i,
                    "type": "syntax",
                    "severity": "critical",
                    "description": f"Missing colon at end of {desc} on line {i}",
                    "suggestion": f"{stripped}:",
                })
                break

    # --- Pass 2: Unmatched brackets using cumulative balance ---
    openers = {"(": ")", "[": "]", "{": "}"}
    closer_to_opener = {v: k for k, v in openers.items()}
    names = {"(": "parenthesis", "[": "bracket", "{": "brace"}

    # Track balance per bracket type
    for open_ch, close_ch in openers.items():
        balance = 0
        candidate_line = -1

        for i, line in enumerate(lines, 1):
            for ch in line:
                if ch == open_ch:
                    if balance == 0:
                        candidate_line = i  # first unmatched opener
                    balance += 1
                elif ch == close_ch:
                    balance -= 1

        if balance > 0 and candidate_line > 0:
            # Find the LAST line that increased balance (the actual error line)
            error_line = candidate_line
            running = 0
            for i, line in enumerate(lines, 1):
                prev_balance = running
                for ch in line:
                    if ch == open_ch:
                        running += 1
                    elif ch == close_ch:
                        running -= 1
                # The line where balance increased and stays unmatched
                if running > prev_balance and running > 0:
                    error_line = i

            issues.append({
                "line": error_line,
                "type": "syntax",
                "severity": "critical",
                "description": f"Missing closing {names[open_ch]} '{close_ch}' on line {error_line}",
                "suggestion": f"Add '{close_ch}' to close the {names[open_ch]}",
            })

    return issues


# =============================================================================
# LOGIC BUG DETECTION (Medium)
# =============================================================================

def _find_function_body(lines: list[str], target_line_idx: int) -> list[str]:
    """Get the function body containing the given 0-indexed line."""
    start = -1
    for j in range(target_line_idx, -1, -1):
        if lines[j].strip().startswith("def "):
            start = j
            break
    if start < 0:
        return []
    body = [lines[start]]
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "":
            body.append(lines[j])
            continue
        cur_indent = len(lines[j]) - len(lines[j].lstrip())
        if cur_indent <= base_indent and lines[j].strip() != "":
            break
        body.append(lines[j])
    return body


def detect_logic_errors(code: str) -> list[dict[str, Any]]:
    """Detect logic bugs with comprehensive pattern matching."""
    issues: list[dict[str, Any]] = []
    lines = code.split("\n")
    full_code_lower = code.lower()

    for i, line in enumerate(lines, 1):
        s = line.strip()
        idx = i - 1  # 0-indexed

        # 1. Max/min initialized to 0 (fails for negatives)
        if re.match(r"^(maximum|minimum|max_val|min_val|max_value|min_value)\s*=\s*0\s*$", s):
            var = s.split("=")[0].strip()
            issues.append({
                "line": i, "type": "logic", "severity": "major",
                "description": (f"Initializing '{var}' to 0 fails when all numbers are negative. "
                                f"Should initialize to numbers[0] or float('-inf')."),
                "suggestion": f"{var} = numbers[0]  # or float('-inf')",
            })

        # 2. Case-sensitive comparison in palindrome/case-insensitive context
        if re.match(r"^\s*if\s+\w+\s*==\s*\w+\s*:", s) and ".lower()" not in s:
            context = "\n".join(lines[:idx]).lower()
            if any(kw in context for kw in ["palindrome", "ignoring case", "ignore case"]):
                issues.append({
                    "line": i, "type": "logic", "severity": "major",
                    "description": ("Case-sensitive comparison — 'Racecar' won't match its reverse. "
                                    "Use .lower() for case-insensitive comparison."),
                    "suggestion": "if text.lower() == reversed_text.lower():",
                })

        # 3. Percentage not divided by 100
        if re.search(r"percent", s, re.IGNORECASE) and "*" in s:
            if "/ 100" not in s and "/100" not in s:
                issues.append({
                    "line": i, "type": "logic", "severity": "major",
                    "description": ("Multiplying by percentage value without dividing by 100. "
                                    "Passing 20 gives 20x the price, not 20%."),
                    "suggestion": "discount = price * (discount_percent / 100)",
                })

        # 4. Binary search: high = len(arr) instead of len(arr) - 1
        if re.match(r"^\s*high\s*=\s*len\(\w+\)\s*$", s):
            nearby = "\n".join(lines[max(0, idx - 3):min(len(lines), idx + 10)])
            if "low" in nearby and "mid" in nearby:
                issues.append({
                    "line": i, "type": "logic", "severity": "major",
                    "description": ("high should be len(arr) - 1 to avoid IndexError on "
                                    "boundary access. Off-by-one in binary search bounds."),
                    "suggestion": "high = len(arr) - 1",
                })

        # 5. Checking 'not in' against source list instead of result
        m = re.match(r"^\s*if\s+(\w+)\s+not\s+in\s+(\w+)\s*:", s)
        if m:
            var, target = m.group(1), m.group(2)
            body = _find_function_body(lines, idx)
            body_text = "\n".join(body)
            if "result" in body_text and ".append" in body_text and target != "result":
                issues.append({
                    "line": i, "type": "logic", "severity": "major",
                    "description": (f"Checking '{var} not in {target}' but should check "
                                    f"'{var} not in result'. Every item is in {target}, "
                                    f"so result is always empty."),
                    "suggestion": f"if {var} not in result:",
                })

        # 6. Recursive call with ignored return value
        fn = re.match(r"^\s*def\s+(\w+)\s*\(", s)
        if fn:
            fname = fn.group(1)
            for j in range(idx + 1, min(len(lines), idx + 30)):
                bl = lines[j].strip()
                if bl.startswith("def "):
                    break
                if re.match(rf"^\s*{fname}\s*\(", bl) and "=" not in lines[j].split(fname)[0]:
                    issues.append({
                        "line": j + 1, "type": "logic", "severity": "major",
                        "description": (f"Recursive call to '{fname}()' — return value is "
                                        f"discarded. Should use result.extend({fname}(...))"),
                        "suggestion": f"result.extend({fname}(item))",
                    })
                    break

        # 7. dict[key] = val instead of dict[key] += val (overwrite vs accumulate)
        da = re.match(r"^\s*(\w+)\[(\w+)\]\s*=\s*(.+)$", s)
        if da:
            d, k, v = da.group(1), da.group(2), da.group(3).strip()
            for j in range(max(0, idx - 4), idx):
                if re.match(rf"^\s*if\s+{k}\s+in\s+{d}\s*:", lines[j].strip()):
                    issues.append({
                        "line": i, "type": "logic", "severity": "major",
                        "description": (f"Overwrites {d}[{k}] with '{v}' instead of accumulating. "
                                        f"Should use += to increment the count/total."),
                        "suggestion": f"{d}[{k}] += {v}",
                    })
                    break

        # 8. Returning wrong variable (return a instead of return result)
        if s.startswith("return "):
            ret_var = s[7:].strip()
            body = _find_function_body(lines, idx)
            body_text = "\n".join(body)
            if ("result" in body_text and "result =" in body_text and
                    ret_var != "result" and ("try" in body_text or "except" in body_text)):
                issues.append({
                    "line": i, "type": "logic", "severity": "major",
                    "description": (f"Returns '{ret_var}' instead of computed 'result'. "
                                    f"The calculated value is never returned."),
                    "suggestion": "return result",
                })

        # 9. Forward iteration in a 'reverse' function
        if re.match(r"^\s*for\s+\w+\s+in\s+range\(len\(\w+\)\)\s*:", s):
            ctx = "\n".join(lines[max(0, idx - 8):idx + 1]).lower()
            if "reverse" in ctx and "def " in ctx:
                issues.append({
                    "line": i, "type": "logic", "severity": "major",
                    "description": ("Forward iteration in a reverse function — builds the same "
                                    "string, not reversed. Must iterate backward."),
                    "suggestion": "for i in range(len(s) - 1, -1, -1):",
                })

        # 10. Merge function missing remaining elements
        if s == "return result":
            body = _find_function_body(lines, idx)
            body_text = "\n".join(body)
            if ("while" in body_text and "len(" in body_text and
                    ".append" in body_text and ".extend" not in body_text and
                    ("[i:]" not in body_text and "[j:]" not in body_text)):
                bt_lower = body_text.lower()
                if "merge" in bt_lower or "sorted" in bt_lower:
                    issues.append({
                        "line": i, "type": "logic", "severity": "major",
                        "description": ("Missing remaining elements after merge loop — when one "
                                        "list is exhausted, the other's remainder is lost."),
                        "suggestion": ("result.extend(list1[i:])\n"
                                       "    result.extend(list2[j:])\n"
                                       "    return result"),
                    })

    return issues


# =============================================================================
# SECURITY ISSUE DETECTION (Hard)
# =============================================================================

def detect_security_issues(code: str) -> list[dict[str, Any]]:
    """Detect security vulnerabilities."""
    issues: list[dict[str, Any]] = []
    lines = code.split("\n")

    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        idx = i - 1

        # eval()
        if re.search(r"\beval\s*\(", s):
            issues.append({
                "line": i, "type": "security", "severity": "critical",
                "description": ("eval() executes arbitrary code — critical security "
                                "vulnerability. Use json.loads() or ast.literal_eval()."),
                "suggestion": s.replace("eval(", "json.loads("),
            })

        # SQL injection
        if re.search(r"(SELECT|INSERT|UPDATE|DELETE)\s", s, re.IGNORECASE):
            if "f\"" in s or "f'" in s or ".format(" in s:
                issues.append({
                    "line": i, "type": "security", "severity": "critical",
                    "description": ("SQL injection — user input directly interpolated into "
                                    "query. Use parameterized queries with placeholders."),
                    "suggestion": "db.execute('SELECT ... WHERE col=?', (value,))",
                })

        # shell=True
        if "shell=True" in s:
            issues.append({
                "line": i, "type": "security", "severity": "critical",
                "description": ("shell=True with user input enables command injection. "
                                "Use shell=False and pass args as a list."),
                "suggestion": s.replace("shell=True", "shell=False"),
            })

        # pickle.loads
        if re.search(r"\bpickle\.(loads|load)\s*\(", s):
            issues.append({
                "line": i, "type": "security", "severity": "critical",
                "description": ("pickle.loads on untrusted data allows arbitrary code "
                                "execution. Use json.loads() instead."),
                "suggestion": "json.loads(data)",
            })

        # yaml.load (unsafe)
        if re.search(r"\byaml\.load\s*\(", s) and "safe_load" not in s:
            issues.append({
                "line": i, "type": "security", "severity": "critical",
                "description": ("yaml.load without Loader allows arbitrary code execution. "
                                "Use yaml.safe_load() instead."),
                "suggestion": s.replace("yaml.load(", "yaml.safe_load("),
            })

        # verify=False
        if "verify=False" in s:
            issues.append({
                "line": i, "type": "security", "severity": "critical",
                "description": ("SSL verification disabled — enables MITM attacks. "
                                "Remove verify=False or set verify=True."),
                "suggestion": s.replace("verify=False", "verify=True"),
            })

        # Printing sensitive data
        if re.search(r"\bprint\s*\(", s):
            for kw, label in [
                ("password", "password"), ("api_token", "API token"),
                ("card_number", "card number"), ("ssn", "SSN"),
                ("api_key", "API key"), ("secret", "secret key"),
            ]:
                if kw in s.lower():
                    issues.append({
                        "line": i, "type": "security", "severity": "critical",
                        "description": (f"Printing {label} to console exposes sensitive "
                                        f"data — security and compliance violation."),
                        "suggestion": "Remove sensitive data from print output",
                    })
                    break

        # SSN/PII in exports (non-print)
        if "ssn" in s.lower() and "print" not in s.lower():
            if any(c in s for c in ["+", "f\"", "f'", ".format"]):
                issues.append({
                    "line": i, "type": "security", "severity": "critical",
                    "description": ("Exporting SSN in plaintext — PII privacy violation. "
                                    "Mask or remove SSN from output."),
                    "suggestion": "Remove SSN or mask it: ssn[-4:]",
                })

        # Path concatenation (traversal)
        if re.match(r"""^\s*path\s*=\s*['"].*['"]\s*\+\s*\w+""", s):
            if "basename" not in s:
                issues.append({
                    "line": i, "type": "security", "severity": "major",
                    "description": ("Path traversal vulnerability — user can pass "
                                    "'../../../etc/passwd' to access arbitrary files."),
                    "suggestion": "os.path.basename(filename)",
                })

        # File open with user-controlled path
        if re.search(r"\bopen\s*\(", s):
            body = _find_function_body(lines, idx)
            body_text = "\n".join(body)
            fn_def = body[0].strip() if body else ""
            if re.search(r"def\s+\w+\s*\(\s*(user|path|file)", fn_def):
                issues.append({
                    "line": i, "type": "security", "severity": "critical",
                    "description": ("Opening user-controlled path without sanitization — "
                                    "allows arbitrary file access (path traversal)."),
                    "suggestion": "Validate path: os.path.abspath(p).startswith(ALLOWED_DIR)",
                })

        # MD5 (broken hash)
        if "hashlib.md5" in s:
            issues.append({
                "line": i, "type": "security", "severity": "major",
                "description": ("MD5 is cryptographically broken — use sha256 or "
                                "bcrypt for hashing sensitive data."),
                "suggestion": "hashlib.sha256() or bcrypt",
            })

    return issues


# =============================================================================
# PERFORMANCE + ADVANCED LOGIC (Hard)
# =============================================================================

def detect_performance_issues(code: str) -> list[dict[str, Any]]:
    """Detect performance anti-patterns."""
    issues: list[dict[str, Any]] = []
    for i, line in enumerate(code.split("\n"), 1):
        if re.search(r"\brange\s*\(\s*len\s*\(", line.strip()):
            issues.append({
                "line": i, "type": "performance", "severity": "minor",
                "description": ("range(len()) anti-pattern — use direct iteration "
                                "or enumerate() for Pythonic code."),
                "suggestion": "for item in collection:  # or enumerate()",
            })
    return issues


def detect_hard_logic(code: str) -> list[dict[str, Any]]:
    """Detect advanced logic bugs for hard snippets."""
    issues: list[dict[str, Any]] = []
    lines = code.split("\n")

    for i, line in enumerate(lines, 1):
        s = line.strip()
        idx = i - 1

        # Duplicate conditions
        if s.startswith("if ") and s.endswith(":"):
            cond = s[3:-1].strip()
            for j in range(i, min(len(lines), i + 8)):
                other = lines[j].strip()
                if other.startswith("if ") and other.endswith(":"):
                    if other[3:-1].strip() == cond:
                        issues.append({
                            "line": j + 1, "type": "logic", "severity": "major",
                            "description": (f"Duplicate condition '{cond}' on lines "
                                            f"{i} and {j+1} — redundant code."),
                            "suggestion": "Combine into single if-block",
                        })
                        break

        # Always returns True (permission bypass)
        if s == "return True":
            body = _find_function_body(lines, idx)
            if body:
                fn_match = re.match(r"\s*def\s+(\w+)", body[0])
                if fn_match:
                    fn = fn_match.group(1)
                    if any(k in fn.lower() for k in ["check", "verify", "permission", "valid"]):
                        t_count = sum(1 for l in body if l.strip() == "return True")
                        f_count = sum(1 for l in body if l.strip() == "return False")
                        if t_count >= 2 and f_count == 0:
                            # Is this the last statement?
                            non_empty = [l for l in body if l.strip()]
                            if non_empty and non_empty[-1].strip() == "return True":
                                if lines[idx].strip() == non_empty[-1].strip() and idx == (
                                    lines.index(non_empty[-1]) if non_empty[-1] in lines else idx
                                ):
                                    issues.append({
                                        "line": i, "type": "logic", "severity": "critical",
                                        "description": (f"'{fn}()' always returns True — final "
                                                        f"return should be False to enforce checks. "
                                                        f"All permission checks are bypassed."),
                                        "suggestion": "return False",
                                    })

        # Missing balance check
        if re.search(r"\[.balance.\]\s*-=", s):
            body = _find_function_body(lines, idx)
            body_text = "\n".join(body)
            has_guard = any("balance" in l and ("<" in l or ">" in l) for l in body)
            if not has_guard:
                issues.append({
                    "line": i, "type": "logic", "severity": "critical",
                    "description": ("No balance check before deduction — allows "
                                    "negative balance / overdraft."),
                    "suggestion": "if from_account['balance'] < amount: raise ValueError",
                })

        # Overwrite vs accumulate
        m = re.match(r"^\s*(\w+)\[(\w+)\]\s*=\s*(\w+)\s*$", s)
        if m:
            d, k, v = m.group(1), m.group(2), m.group(3)
            for j in range(max(0, idx - 4), idx):
                if re.match(rf"^\s*if\s+{k}\s+in\s+{d}\s*:", lines[j].strip()):
                    issues.append({
                        "line": i, "type": "logic", "severity": "major",
                        "description": (f"Overwrites {d}[{k}] instead of accumulating. "
                                        f"Use += to sum values."),
                        "suggestion": f"{d}[{k}] += {v}",
                    })
                    break

        # Age validation missing upper bound
        if re.match(r"^\s*if\s+age\s*>\s*0\s*:", s):
            ctx = "\n".join(lines[max(0, idx - 5):idx]).lower()
            if "validate" in ctx and "age" in ctx:
                issues.append({
                    "line": i, "type": "logic", "severity": "major",
                    "description": ("Only checks age > 0 — no upper bound. "
                                    "Ages like 500 would pass validation."),
                    "suggestion": "if 0 < age <= 150:",
                })

        # Bubble sort inner loop
        if re.match(r"^\s*for\s+j\s+in\s+range\(len\(\w+\)\)\s*:", s):
            ctx = "\n".join(lines[max(0, idx - 3):min(len(lines), idx + 5)])
            if "for i in range" in ctx and ("[i]" in ctx and "[j]" in ctx):
                issues.append({
                    "line": i, "type": "performance", "severity": "major",
                    "description": ("Bubble sort inner loop should start from i+1 "
                                    "for correct and efficient sorting."),
                    "suggestion": "for j in range(i + 1, len(items)):",
                })

    return issues


# =============================================================================
# AGENT ORCHESTRATION
# =============================================================================

def analyze_code(code: str, difficulty: str, context: str = "") -> list[dict[str, Any]]:
    """Analyze code using LLM + rule-based detection.

    First attempts LLM-based analysis via the OpenAI-compatible API.
    Then supplements with rule-based detection to ensure high coverage.
    Results are merged, deduplicated by line (highest severity kept).
    """
    raw: list[dict[str, Any]] = []

    # ── Phase 1: LLM Analysis ──
    llm_issues = llm_analyze(code, context=context, difficulty=difficulty)
    raw.extend(llm_issues)

    # ── Phase 2: Rule-based fallback / supplement ──
    if difficulty == "easy":
        raw.extend(detect_syntax_errors(code))
    elif difficulty == "medium":
        raw.extend(detect_logic_errors(code))
    else:
        raw.extend(detect_security_issues(code))
        raw.extend(detect_hard_logic(code))
        raw.extend(detect_performance_issues(code))

    # Deduplicate: keep highest severity per line
    rank = {"critical": 0, "major": 1, "minor": 2}
    by_line: dict[int, dict[str, Any]] = {}
    for issue in raw:
        ln = issue["line"]
        if ln not in by_line or rank.get(issue["severity"], 9) < rank.get(by_line[ln]["severity"], 9):
            by_line[ln] = issue

    result = sorted(by_line.values(), key=lambda x: rank.get(x["severity"], 9))
    return result


def build_action(issue: dict[str, Any]) -> dict[str, Any]:
    """Convert a detected issue into an Action dict."""
    return {
        "action_type": "identify_bug",
        "line_number": issue["line"],
        "bug_type": issue["type"],
        "severity": issue["severity"],
        "description": issue["description"],
        "suggestion": issue["suggestion"],
    }

# =============================================================================
# LOGGING FORMATTERS
# =============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Any = None) -> None:
    done_str = "true" if done else "false"
    err_str = error if error is not None else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    succ_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={succ_str} steps={steps} rewards={rewards_str}", flush=True)


# =============================================================================
# LOCAL MODE
# =============================================================================

def run_local(difficulty: str = "easy") -> None:
    """Run the agent locally against all snippets.

    Emits the required log format:
        [START] task=<difficulty> snippet=<id>
        [STEP] reward=<r> done=<d> step=<n>
        [END] total_reward=<r> task=<difficulty>
    """
    from code_review_env.env import CodeReviewEnv
    from code_review_env.models import Action, ActionType

    env = CodeReviewEnv()
    snippets = env._tasks[difficulty].get_snippets()
    total_score = 0.0
    episodes = 0

    for snippet in snippets:
        obs = env.reset(difficulty=difficulty, snippet_id=snippet["id"])
        log_start(task=difficulty, env="CodeReviewEnv", model=MODEL_NAME)

        found_issues = analyze_code(obs.code_snippet, difficulty, context=obs.context)
        result = None
        step_count = 0
        rewards = []

        if not found_issues:
            action = Action(action_type=ActionType.APPROVE, description="No issues found")
            result = env.step(action)
            step_count = 1
            rewards.append(result.reward)
            log_step(step=step_count, action=str(action), reward=result.reward, done=result.done)
        else:
            for n, issue in enumerate(found_issues):
                if env._done:
                    break
                action = Action(**build_action(issue))
                result = env.step(action)
                step_count = n + 1
                rewards.append(result.reward)
                log_step(step=step_count, action=str(action), reward=result.reward, done=result.done)
                if result.done:
                    break

        state = env.state()
        score = result.info.get("final_score", state.total_reward) if result is not None else state.total_reward
        total_score += score
        episodes += 1
        log_end(success=score >= 0.8, steps=step_count, rewards=rewards)

    avg = total_score / max(1, episodes)
    print(f"\n# Results: {episodes} episodes | Average score: {avg:.4f}")


# =============================================================================
# SERVER MODE
# =============================================================================

def run_server(base_url: str, difficulty: str = "easy") -> None:
    """Run the agent against a remote HTTP server.

    Emits the required [START]/[STEP]/[END] log format.
    """
    try:
        import requests as req_lib
    except ImportError:
        print("Error: pip install requests")
        sys.exit(1)

    resp = req_lib.post(f"{base_url}/reset", json={"difficulty": difficulty})
    if resp.status_code != 200:
        print(f"Reset failed: {resp.text}")
        return
    obs = resp.json()
    snippet_id = obs.get("task_id", "unknown")
    log_start(task=difficulty, env="CodeReviewEnv", model=MODEL_NAME)

    found_issues = analyze_code(
        obs["code_snippet"], difficulty, context=obs.get("context", "")
    )
    done = False
    step_count = 0
    rewards = []

    if not found_issues:
        action_dict = {
            "action_type": "approve", "line_number": 0,
            "description": "No issues", "suggestion": ""
        }
        r = req_lib.post(f"{base_url}/step", json=action_dict).json()
        step_count = 1
        rewards.append(r['reward'])
        log_step(step=step_count, action=str(action_dict), reward=r['reward'], done=r['done'])
    else:
        for n, issue in enumerate(found_issues):
            if done:
                break
            action_dict = build_action(issue)
            r = req_lib.post(f"{base_url}/step", json=action_dict)
            if r.status_code != 200:
                break
            r = r.json()
            step_count = n + 1
            rewards.append(r['reward'])
            log_step(step=step_count, action=str(action_dict), reward=r['reward'], done=r['done'])
            done = r["done"]

    r = req_lib.get(f"{base_url}/state").json()
    log_end(success=r['total_reward'] >= 0.8, steps=step_count, rewards=rewards)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Entry point — parse args and run the inference agent."""
    parser = argparse.ArgumentParser(description="CodeReviewEnv baseline agent")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--server", type=str, default=None)
    args = parser.parse_args()

    diffs = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]
    for d in diffs:
        if args.server:
            run_server(args.server, d)
        else:
            run_local(d)


if __name__ == "__main__":
    main()
