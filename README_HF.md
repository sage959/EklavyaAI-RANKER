---
title: Eklavya AI Ranking Sandbox
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.52.2"
app_file: sandbox_demo.py
pinned: true
license: mit
---

# 🧠 Eklavya AI — Ranking Sandbox

End-to-end candidate ranking pipeline demo for the **Redrob AI Hiring Hackathon 2026**.

## What it does

Runs our full ranking pipeline on up to 100 candidates:
- **BM25 lexical retrieval** (always active)
- **FAISS semantic retrieval** (optional toggle)
- **Cross-encoder reranking** (optional toggle)
- **Feature engineering** (relevance, production depth, technical alignment, behavioral signals)
- **XGBoost blended scoring**
- **Proof-of-work, credibility, and contradiction checks**
- **Ranked CSV output** in the exact submission format

## How to use

1. Select the **pre-loaded sample** (50 candidates) or upload a custom JSONL/JSON file
2. Click **▶ Run Ranking**
3. View the ranked results table
4. Download the `TIER3.csv`

## Performance

- **50 candidates**: ~8 seconds on CPU
- **100 candidates**: ~15 seconds on CPU
- Well within the 5-minute budget
