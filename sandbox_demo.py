"""
sandbox_demo.py — Streamlit sandbox demo for Eklavya AI Ranking Engine.

A self-contained, CPU-only demo that runs the full ranking pipeline end-to-end
on up to 100 candidates.  Designed for hackathon submission verification on
HuggingFace Spaces or Streamlit Cloud.

Usage:
    streamlit run sandbox_demo.py
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path

import streamlit as st

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Project root (works both locally and on HF Spaces) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eklavya AI — Ranking Sandbox",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* -- Global theme -- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* -- Header banner -- */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .hero-banner h1 {
        margin: 0 0 0.3rem 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }

    /* -- Stat cards -- */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .stat-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.65;
        margin-top: 0.3rem;
    }

    /* -- Pipeline stage badges -- */
    .stage-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        margin: 0.2rem;
        font-size: 0.78rem;
        color: #a0b4ff;
    }
    .stage-badge.done {
        background: rgba(72, 199, 142, 0.15);
        border-color: rgba(72, 199, 142, 0.3);
        color: #48c78e;
    }

    /* -- Results table -- */
    .results-table {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* -- Footer -- */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.75rem;
        margin-top: 3rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🧠 Eklavya AI — Ranking Sandbox</h1>
    <p>End-to-end candidate ranking pipeline demo &middot; Hybrid BM25 + Semantic retrieval &middot;
       Cross-encoder reranking &middot; XGBoost blending &middot; CPU only</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")

    # JD text
    default_jd = (PROJECT_ROOT / "ranking" / "pipeline.py").exists()
    if default_jd:
        from ranking.pipeline import DEFAULT_JD_TEXT
        jd_text = st.text_area(
            "Job Description",
            value=DEFAULT_JD_TEXT.strip(),
            height=200,
            help="The JD used to score candidates. Edit to test different roles.",
        )
    else:
        jd_text = st.text_area(
            "Job Description",
            value="Senior AI Engineer — retrieval, ranking, search, Python, FAISS, ML",
            height=200,
        )

    use_semantic = st.toggle("Semantic Retrieval (FAISS)", value=False,
                             help="Disable for faster CPU-only runs. BM25 lexical retrieval is always active.")
    use_reranker = st.toggle("Cross-Encoder Reranking", value=False,
                             help="Adds a cross-encoder reranking step. Slower but more precise.")
    top_k = st.slider("Top K results", min_value=10, max_value=100, value=100, step=10)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.75rem;'>"
        "Eklavya AI v1.0 &middot; Redrob Hackathon 2026</div>",
        unsafe_allow_html=True,
    )


# ── Data input ───────────────────────────────────────────────────────────────
st.markdown("### 📂 Input Candidates")

input_mode = st.radio(
    "Choose input source",
    ["Pre-loaded sample (50 candidates)", "Upload custom file"],
    horizontal=True,
    label_visibility="collapsed",
)

candidates_raw: list[dict] = []

if input_mode == "Pre-loaded sample (50 candidates)":
    sample_path = PROJECT_ROOT / "data" / "sample_candidates.jsonl"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        candidates_raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        st.success(f"Loaded **{len(candidates_raw)}** candidates from pre-loaded sample.")
    else:
        st.error(f"Sample file not found at `{sample_path}`. Please create it first.")

else:
    uploaded = st.file_uploader(
        "Upload a JSON array or JSONL file (max 100 candidates)",
        type=["json", "jsonl"],
        help="Each candidate should follow the candidate_schema with fields: "
             "candidate_id, profile, career_history, skills, redrob_signals, etc.",
    )
    if uploaded is not None:
        content = uploaded.read().decode("utf-8")
        # Try JSON array first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                candidates_raw = parsed[:100]
            elif isinstance(parsed, dict):
                candidates_raw = [parsed]
        except json.JSONDecodeError:
            # Try JSONL
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        candidates_raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            candidates_raw = candidates_raw[:100]

        if candidates_raw:
            st.success(f"Parsed **{len(candidates_raw)}** candidates from upload.")
        else:
            st.error("Could not parse any candidates from the uploaded file.")


# ── Validation ───────────────────────────────────────────────────────────────
valid_candidates = [c for c in candidates_raw if c.get("candidate_id")]
if candidates_raw and not valid_candidates:
    st.warning(
        "No candidates have a `candidate_id` field. "
        "Make sure your file follows the hackathon candidate schema."
    )

# ── Run pipeline ─────────────────────────────────────────────────────────────
st.markdown("### 🚀 Run Pipeline")

col_run, col_info = st.columns([1, 3])
with col_run:
    run_clicked = st.button(
        "▶ Run Ranking",
        type="primary",
        disabled=len(valid_candidates) == 0,
    )
with col_info:
    if valid_candidates:
        st.info(
            f"Ready: **{len(valid_candidates)}** candidates  ·  "
            f"Semantic: {'ON' if use_semantic else 'OFF'}  ·  "
            f"Reranker: {'ON' if use_reranker else 'OFF'}  ·  "
            f"Top K: {top_k}"
        )
    else:
        st.warning("Load or upload candidates first.")


if run_clicked and valid_candidates:
    # -- Pipeline stages with progress reporting --
    progress = st.progress(0, text="Initializing pipeline...")
    status_container = st.container()
    stage_times: dict[str, float] = {}

    try:
        # Stage 1: Initialize pipeline
        t_start = time.time()
        progress.progress(5, text="Stage 1/6: Initializing pipeline...")

        from ranking.pipeline import RankingPipeline

        pipeline = RankingPipeline(
            jd_text=jd_text,
            use_semantic=use_semantic,
            use_reranker=use_reranker,
        )
        stage_times["init"] = time.time() - t_start

        # Stage 2: Load candidates
        progress.progress(15, text="Stage 2/6: Loading candidates...")
        t2 = time.time()
        count = pipeline.load_candidates_from_list(valid_candidates)
        stage_times["load"] = time.time() - t2

        # Stage 3: Build index
        progress.progress(30, text="Stage 3/6: Building BM25 index...")
        t3 = time.time()
        index_info = pipeline.build_index()
        stage_times["index"] = time.time() - t3

        # Stage 4: Rank
        progress.progress(55, text="Stage 4/6: Running hybrid retrieval + scoring...")
        t4 = time.time()
        results = pipeline.rank(top_k=min(top_k, count))
        stage_times["rank"] = time.time() - t4

        # Stage 5: Export CSV
        progress.progress(85, text="Stage 5/6: Generating CSV...")
        t5 = time.time()

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results[:100]:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                r["final_score"],
                r["reasoning_text"],
            ])
        csv_output = csv_buffer.getvalue()
        stage_times["export"] = time.time() - t5

        # Stage 6: Display
        progress.progress(95, text="Stage 6/6: Rendering results...")

        total_time = time.time() - t_start
        stage_times["total"] = total_time

        progress.progress(100, text="Pipeline complete!")
        time.sleep(0.3)
        progress.empty()

        # ── Stats row ────────────────────────────────────────────────────
        st.markdown("### 📊 Results")

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">Candidates Ranked</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_time:.1f}s</div>
                <div class="stat-label">Total Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{results[0]['final_score']:.4f}</div>
                <div class="stat-label">Top Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{index_info.get('bm25_docs', 0)}</div>
                <div class="stat-label">BM25 Documents</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Pipeline stage timings ───────────────────────────────────────
        with st.expander("Pipeline Stage Timings", expanded=False):
            stages = [
                ("Initialization (JD parse + XGBoost load)", "init"),
                ("Candidate loading + profiling", "load"),
                ("BM25 + semantic index build", "index"),
                ("Hybrid retrieval + scoring", "rank"),
                ("CSV export", "export"),
            ]
            for label, key in stages:
                t = stage_times.get(key, 0)
                pct = (t / total_time * 100) if total_time > 0 else 0
                st.markdown(
                    f'<span class="stage-badge done">{label}: {t:.2f}s ({pct:.0f}%)</span>',
                    unsafe_allow_html=True,
                )
            st.caption(f"Total wall-clock: **{total_time:.2f}s**  ·  "
                       f"5-minute budget: **{total_time/300*100:.1f}%** used")

        # ── Results table ────────────────────────────────────────────────
        import pandas as pd

        df = pd.DataFrame([
            {
                "Rank": r["rank"],
                "Candidate ID": r["candidate_id"],
                "Name": r.get("name", ""),
                "Title": r.get("current_title", ""),
                "Score": round(r["final_score"], 4),
                "Rule Score": round(r.get("rule_score", r["final_score"]), 4),
                "XGB Score": round(r["xgb_score"], 4) if r.get("xgb_score") is not None else "N/A",
                "PoW": round(r.get("pow_score", 0), 3),
                "Contra": round(r.get("contra_penalty", 1.0), 3),
                "Reasoning": r.get("reasoning_text", "")[:150],
            }
            for r in results
        ])

        st.dataframe(
            df,
            width="stretch",
            height=min(600, 35 * len(results) + 38),
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Score": st.column_config.NumberColumn(format="%.4f"),
                "Rule Score": st.column_config.NumberColumn(format="%.4f"),
                "PoW": st.column_config.NumberColumn(format="%.3f"),
                "Contra": st.column_config.NumberColumn(format="%.3f"),
                "Reasoning": st.column_config.TextColumn(width="large"),
            },
        )

        # ── Download button ──────────────────────────────────────────────
        st.download_button(
            label="📥 Download Ranked CSV (TIER3.csv)",
            data=csv_output,
            file_name="TIER3.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )

        # ── Detailed JSON export ─────────────────────────────────────────
        detailed_json = json.dumps(
            [{
                "candidate_id": r["candidate_id"],
                "rank": r["rank"],
                "name": r.get("name", ""),
                "current_title": r.get("current_title", ""),
                "final_score": r["final_score"],
                "rule_score": r.get("rule_score", r["final_score"]),
                "xgb_score": r.get("xgb_score"),
                "legacy_score": r.get("legacy_score", 0),
                "retrieval_score": r.get("retrieval_score", 0),
                "pow_score": r.get("pow_score", 0),
                "contra_penalty": r.get("contra_penalty", 1.0),
                "credibility_penalty": r.get("credibility_penalty", 0.0),
                "reasoning": r.get("reasoning_text", ""),
            } for r in results],
            indent=2,
        )

        st.download_button(
            label="📥 Download Detailed JSON",
            data=detailed_json,
            file_name="detailed_results.json",
            mime="application/json",
        )

        # ── Top 3 deep dive ──────────────────────────────────────────────
        with st.expander("Top 3 Candidate Details", expanded=True):
            for r in results[:3]:
                st.markdown(f"""
                **#{r['rank']} — {r.get('name', r['candidate_id'])}**
                ({r.get('current_title', 'N/A')} · Score: {r['final_score']:.4f})

                > {r.get('reasoning_text', 'No reasoning available.')}
                """)
                st.markdown("---")

    except Exception as e:
        progress.empty()
        st.error(f"Pipeline failed: {e}")
        logger.exception("Pipeline execution failed")
        import traceback
        st.code(traceback.format_exc(), language="python")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Eklavya AI Ranking Engine v1.0 &middot; Redrob AI Hiring Hackathon 2026<br>
    Hybrid BM25 + FAISS retrieval &middot; Cross-encoder reranking &middot;
    XGBoost blending &middot; Proof-of-work &middot; Credibility checks &middot; Contradiction detection
</div>
""", unsafe_allow_html=True)
