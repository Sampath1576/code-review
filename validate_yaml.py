"""Validate openenv.yaml structure."""
import yaml

with open("openenv.yaml") as f:
    data = yaml.safe_load(f)

required_fields = ["name", "version", "spec_version", "type", "runtime", "app", "port"]
missing = [f for f in required_fields if f not in data]

print("=" * 50)
print("OpenEnv YAML Validation")
print("=" * 50)

if missing:
    print(f"FAIL: Missing fields: {missing}")
else:
    print("PASS: All required fields present")

print(f"  name: {data.get('name')}")
print(f"  version: {data.get('version')}")
print(f"  spec_version: {data.get('spec_version')}")
print(f"  type: {data.get('type')}")
print(f"  runtime: {data.get('runtime')}")
print(f"  app: {data.get('app')}")
print(f"  port: {data.get('port')}")

tasks = data.get("tasks", [])
print(f"\n  Tasks: {len(tasks)} defined")
for t in tasks:
    print(f"    - {t['name']}: {t.get('snippets', '?')} snippets, max {t.get('max_steps', '?')} steps")

env_vars = data.get("env_vars", [])
print(f"\n  Env Vars: {len(env_vars)} defined")
for v in env_vars:
    print(f"    - {v['name']}: {v.get('description', '')[:60]}")

baseline = data.get("baseline", {})
if baseline:
    print(f"\n  Baseline agent: {baseline.get('agent')}")
    scores = baseline.get("scores", {})
    for k, v in scores.items():
        print(f"    {k}: {v}")

print("\n" + "=" * 50)
print("VALIDATION: PASSED" if not missing else "VALIDATION: FAILED")
print("=" * 50)
