"""Test custom code analysis endpoint."""
import requests

code = """import subprocess

def run_cmd(user_input):
    result = subprocess.call(user_input, shell=True)
    return result

def load(raw):
    return eval(raw)

def auth(username, password, db):
    query = f"SELECT * FROM users WHERE name='{username}'"
    return db.execute(query)
"""

r = requests.post("http://localhost:7860/api/analyze-custom", json={"code": code})
data = r.json()
print(f"Status: {r.status_code}")
print(f"Total issues: {data['total_issues']}")
print(f"Summary: {data['summary']}")
print(f"Steps: {len(data['steps'])}")
for s in data['steps']:
    print(f"  [{s['bug_type']}] line {s['line_number']} ({s['severity']}): {s['description'][:80]}")
