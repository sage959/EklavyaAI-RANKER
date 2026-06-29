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

def submit_and_wait(jd_text: str):
    req = urllib.request.Request(
        f"{BASE}/rank-custom-jd",
        data=json.dumps({"jd_text": jd_text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    task_id = resp["task_id"]
    while True:
        poll = json.loads(urllib.request.urlopen(f"{BASE}/task-status/{task_id}").read())
        status = poll["status"]
        if status == "done":
            return poll["result"]["results"]
        elif status == "error":
            print("ERROR", poll)
            return []
        time.sleep(2)

print("Run 1...")
r1 = submit_and_wait(CUSTOM_JD)
d1 = {r['candidate_id']: r for r in r1}

print("Run 2...")
r2 = submit_and_wait(CUSTOM_JD)
d2 = {r['candidate_id']: r for r in r2}

print("Diffing sub-scores...")
for cid in d1:
    if cid in d2:
        c1 = d1[cid]
        c2 = d2[cid]
        if c1['final_score'] != c2['final_score']:
            print(f"Diff for {cid}:")
            for k in ['final_score', 'rule_score', 'xgb_score', 'legacy_score', 'retrieval_score', 'pow_score', 'behavioral_score', 'contra_penalty', 'credibility_penalty']:
                v1 = c1.get(k)
                v2 = c2.get(k)
                if v1 != v2:
                    print(f"  {k}: {v1} != {v2}")
