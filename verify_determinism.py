"""
Determinism verification — uses the running API server.
Sends the same custom JD twice and compares the ranked results.
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

CUSTOM_JD = """Data Scientist

Requirements:
- 3+ years Python experience
- Experience with ML pipelines and model deployment
- Strong SQL and data analysis skills
- Familiarity with deep learning frameworks

Nice to have:
- Cloud experience (AWS, GCP)
- NLP expertise
- Experience with A/B testing

Responsibilities:
- Build and deploy ML models
- Analyze data to drive product decisions
- Collaborate with engineering teams
"""


def submit_and_wait(jd_text: str, label: str):
    """Submit a custom JD ranking and poll until done."""
    # Submit
    req = urllib.request.Request(
        f"{BASE}/rank-custom-jd",
        data=json.dumps({"jd_text": jd_text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    task_id = resp["task_id"]
    print(f"  [{label}] Task submitted: {task_id}")

    # Poll
    while True:
        poll = json.loads(
            urllib.request.urlopen(f"{BASE}/task-status/{task_id}").read()
        )
        status = poll["status"]
        if status == "done":
            results = poll["result"]["results"]
            return [(r["candidate_id"], r["final_score"], r["rank"]) for r in results[:10]]
        elif status == "error":
            print(f"  [{label}] ERROR: {poll.get('error')}")
            return []
        time.sleep(2)


print("=" * 60)
print("DETERMINISM VERIFICATION (via API)")
print("=" * 60)

print("\nRun 1...")
run1 = submit_and_wait(CUSTOM_JD, "Run1")
print("Run 1 top 10:")
for cid, score, rank in run1:
    print(f"  #{rank} {cid} score={score}")

# Small pause between runs
time.sleep(1)

print("\nRun 2...")
run2 = submit_and_wait(CUSTOM_JD, "Run2")
print("Run 2 top 10:")
for cid, score, rank in run2:
    print(f"  #{rank} {cid} score={score}")

print("\n" + "=" * 60)
if run1 == run2:
    print("PASS: Both runs produced IDENTICAL top-10 results.")
else:
    print("FAIL: Results differ between runs!")
    for i, (r1, r2) in enumerate(zip(run1, run2)):
        if r1 != r2:
            print(f"  Diff at position {i}: {r1} != {r2}")
print("=" * 60)
