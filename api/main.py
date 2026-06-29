"""
api/main.py — FastAPI service layer for the Eklavya AI ranking engine.
Exposes endpoints for JD parsing, candidate indexing, ranking, and health checks.
"""
from __future__ import annotations

import json
import logging
import time
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ranking.pipeline import RankingPipeline, DEFAULT_JD_TEXT
from ranking.jd_parser import parse_jd
from ranking.resume_parser import parse_resume_file, fetch_drive_folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Eklavya AI — Ranking Engine API",
    description="Proof-of-work candidate ranking system with hybrid retrieval, "
                "cross-encoder reranking, and explainable scoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (initialised on first use)
_pipeline: Optional[RankingPipeline] = None

# Background tasks registry for long-running pipeline operations
_tasks: dict[str, dict[str, Any]] = {}


@app.on_event("startup")
def _startup_preload():
    """Pre-load pipeline indices and ML models at server start.

    This eliminates the ~330s cold-start penalty on the first custom JD
    request.  Models are loaded in the main thread where native C++
    libraries (FAISS, PyTorch) are stable.
    """
    global _pipeline
    logger.info("=" * 60)
    logger.info("SERVER STARTUP: Pre-loading pipeline and models...")
    logger.info("=" * 60)

    t0 = time.time()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    # 1. Create pipeline and load pre-computed indices (BM25 + FAISS)
    _pipeline = RankingPipeline(use_semantic=True, use_reranker=True)

    if (data_dir / "bm25_index.pkl").exists():
        index_info = _pipeline.load_precomputed_index(str(data_dir))
        logger.info(f"Indices loaded: {index_info}")
    else:
        logger.warning("Pre-computed indices not found. Pipeline will init on first use.")

    # 2. Warm up sentence-transformers model (used by FAISS semantic query)
    try:
        from ranking.retrieval_semantic import _load_model
        _load_model()
        logger.info("Sentence-transformers model pre-loaded.")
    except Exception as e:
        logger.warning(f"Could not pre-load sentence-transformers: {e}")

    # 3. Warm up cross-encoder model (used by reranker)
    try:
        from ranking.rerank import _load_cross_encoder
        _load_cross_encoder()
        logger.info("Cross-encoder model pre-loaded.")
    except Exception as e:
        logger.warning(f"Could not pre-load cross-encoder: {e}")

    elapsed = time.time() - t0
    logger.info(f"Startup pre-load complete in {elapsed:.1f}s")
    logger.info("=" * 60)


def _get_pipeline() -> RankingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RankingPipeline(use_semantic=True, use_reranker=True)
    return _pipeline


# ── Pydantic models ──────────────────────────────────────────────────────────

class ParseJDRequest(BaseModel):
    """Request body for JD parsing."""
    jd_text: str = Field(..., description="Raw job description text")

class ParseJDResponse(BaseModel):
    """Parsed JD response."""
    role_title: str
    seniority: str
    min_experience_years: float
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    responsibilities: list[str]
    domains: list[str]
    intent_facets: dict[str, float]
    ranking_priorities: list[str]

class IndexCandidatesRequest(BaseModel):
    """Request to index candidates from a file path."""
    candidates_path: str = Field(..., description="Path to candidates.jsonl file")
    limit: Optional[int] = Field(None, description="Max candidates to load")
    jd_text: Optional[str] = Field(None, description="Custom JD text (optional)")

class IndexResponse(BaseModel):
    """Index build response."""
    status: str
    candidate_count: int
    bm25_docs: int
    semantic_available: bool
    total_time_s: float

class RankRequest(BaseModel):
    """Request to run ranking on indexed candidates."""
    top_k: int = Field(100, description="Number of candidates to rank")
    output_csv: str = Field("output/TIER3.csv", description="CSV output path")
    output_json: str = Field("output/detailed_results.json", description="JSON output path")

class RankFromFilesRequest(BaseModel):
    """Run full pipeline from file paths."""
    candidates_path: str = Field(..., description="Path to candidates.jsonl")
    jd_text: Optional[str] = Field(None, description="Custom JD text")
    limit: Optional[int] = Field(None, description="Max candidates to load")
    top_k: int = Field(100, description="Number of candidates to rank")
    output_csv: str = Field("output/TIER3.csv", description="CSV output path")
    output_json: str = Field("output/detailed_results.json", description="JSON output path")
    use_semantic: bool = Field(True, description="Use semantic retrieval")
    use_reranker: bool = Field(True, description="Use cross-encoder reranking")

class RankResult(BaseModel):
    """Single ranked candidate."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str

class RankResponse(BaseModel):
    """Ranking response."""
    status: str
    ranked_count: int
    csv_path: str
    json_path: str
    top_results: list[RankResult]
    elapsed_s: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pipeline_loaded: bool
    candidates_indexed: int
    semantic_available: bool

class RankWithCustomJDRequest(BaseModel):
    """Request to re-score existing results against a custom JD."""
    jd_text: str = Field(..., description="Raw custom job description text")

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint."""
    pipeline = _get_pipeline()
    return HealthResponse(
        status="ok",
        pipeline_loaded=pipeline._indexed,
        candidates_indexed=len(pipeline.profiles),
        semantic_available=pipeline.semantic_index.available,
    )


@app.post("/parse-jd", response_model=ParseJDResponse, tags=["JD"])
def parse_job_description(req: ParseJDRequest):
    """Parse a raw job description into a structured schema."""
    parsed = parse_jd(req.jd_text)
    return ParseJDResponse(
        role_title=parsed.role_title,
        seniority=parsed.seniority,
        min_experience_years=parsed.min_experience_years,
        must_have_skills=parsed.must_have_skills,
        nice_to_have_skills=parsed.nice_to_have_skills,
        responsibilities=parsed.responsibilities,
        domains=parsed.domains,
        intent_facets=parsed.intent_facets,
        ranking_priorities=parsed.ranking_priorities,
    )


@app.post("/index-candidates", response_model=IndexResponse, tags=["Index"])
def index_candidates(req: IndexCandidatesRequest):
    """Load and index candidates from a JSONL file."""
    global _pipeline

    jd_text = req.jd_text or DEFAULT_JD_TEXT
    _pipeline = RankingPipeline(
        jd_text=jd_text,
        use_semantic=True,
        use_reranker=True,
    )

    try:
        count = _pipeline.load_candidates_from_jsonl(req.candidates_path, limit=req.limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    index_info = _pipeline.build_index()

    return IndexResponse(
        status=index_info.get("status", "ok"),
        candidate_count=index_info.get("candidate_count", count),
        bm25_docs=index_info.get("bm25_docs", 0),
        semantic_available=index_info.get("semantic_available", False),
        total_time_s=index_info.get("total_time_s", 0),
    )


@app.post("/rank", response_model=RankResponse, tags=["Ranking"])
def rank_candidates(req: RankRequest):
    """Rank previously indexed candidates."""
    pipeline = _get_pipeline()
    if not pipeline._indexed:
        raise HTTPException(status_code=400, detail="No candidates indexed. Call /index-candidates first.")

    t0 = time.time()
    results = pipeline.rank(top_k=req.top_k)

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / req.output_csv
    json_path = project_root / req.output_json

    pipeline.export_submission_csv(results, str(csv_path))
    pipeline.export_detailed_json(results, str(json_path))

    elapsed = time.time() - t0

    top_results = [
        RankResult(
            candidate_id=r["candidate_id"],
            rank=r["rank"],
            score=r["final_score"],
            reasoning=r["reasoning_text"],
        )
        for r in results[:10]
    ]

    return RankResponse(
        status="ok",
        ranked_count=len(results),
        csv_path=str(csv_path),
        json_path=str(json_path),
        top_results=top_results,
        elapsed_s=round(elapsed, 2),
    )


@app.post("/rank-from-files", response_model=RankResponse, tags=["Ranking"])
def rank_from_files(req: RankFromFilesRequest):
    """Full pipeline: load → index → rank → export from file paths."""
    global _pipeline

    t0 = time.time()
    jd_text = req.jd_text or DEFAULT_JD_TEXT

    _pipeline = RankingPipeline(
        jd_text=jd_text,
        use_semantic=req.use_semantic,
        use_reranker=req.use_reranker,
    )

    try:
        _pipeline.load_candidates_from_jsonl(req.candidates_path, limit=req.limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    _pipeline.build_index()
    results = _pipeline.rank(top_k=req.top_k)

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / req.output_csv
    json_path = project_root / req.output_json

    _pipeline.export_submission_csv(results, str(csv_path))
    _pipeline.export_detailed_json(results, str(json_path))

    elapsed = time.time() - t0

    top_results = [
        RankResult(
            candidate_id=r["candidate_id"],
            rank=r["rank"],
            score=r["final_score"],
            reasoning=r["reasoning_text"],
        )
        for r in results[:10]
    ]

    return RankResponse(
        status="ok",
        ranked_count=len(results),
        csv_path=str(csv_path),
        json_path=str(json_path),
        top_results=top_results,
        elapsed_s=round(elapsed, 2),
    )

import re as _re


def _pipeline_results_to_frontend(pipeline: RankingPipeline,
                                  results: list[dict]) -> list[dict]:
    """Convert RankingPipeline.rank() output to the frontend JSON format.

    This is a shared helper so that every endpoint returns the same
    structure regardless of how the pipeline was invoked.
    """
    export_data = []
    for r in results:
        cid = r["candidate_id"]
        profile = pipeline.profiles.get(cid)

        entry = {
            "candidate_id": cid,
            "rank": r["rank"],
            "name": r.get("name", ""),
            "current_title": r.get("current_title", ""),
            "final_score": r["final_score"],
            "rule_score": r.get("rule_score", r["final_score"]),
            "xgb_score": r.get("xgb_score"),
            "legacy_score": r.get("legacy_score", 0),
            "retrieval_score": r.get("retrieval_score", 0),
            "pow_score": r.get("pow_score", 0),
            "behavioral_score": r.get("behavioral_score", 0.5),
            "contra_penalty": r.get("contra_penalty", 1.0),
            "credibility_penalty": r.get("credibility_penalty", 0.0),
            "reasoning": r.get("reasoning_text", ""),
            "explanation": r.get("explanation", {}),
            "pow_detail": r.get("pow_detail", {}),
            "contradiction_detail": r.get("contradiction_detail", {}),
            "credibility_detail": r.get("credibility_detail", {}),
        }

        if profile:
            entry.update({
                "skills": [s.name for s in profile.skills],
                "location": profile.location,
                "country": profile.country,
                "experience": f"{profile.years_of_experience:.1f} yrs",
                "current_company": profile.current_company,
                "education": ", ".join([
                    f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
                    for edu in profile.education if edu
                ]),
                "careerHistory": [
                    {
                        "role": e.title,
                        "company": e.company,
                        "period": f"{e.start_date} - {e.end_date or 'Present'}",
                        "note": e.description,
                    }
                    for e in profile.career_events
                ],
                "redrob_signals": profile.redrob_signals,
            })

        export_data.append(entry)

    return export_data


def _run_custom_jd_task(task_id: str, jd_text: str):
    """Background worker: runs the full ranking pipeline for a custom JD.

    Updates _tasks[task_id] with progress, and stores the final result
    or error when complete.
    """
    try:
        _tasks[task_id]["status"] = "loading_indices"
        t0 = time.time()

        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"

        # 1. Normalize line endings (fixes truncation differences for \r\n vs \n)
        jd_text = jd_text.replace('\r\n', '\n')
        
        # 2. Parse the custom JD (used ONLY for UI display, not scoring)
        parsed = parse_jd(jd_text)

        # 3. Safely get the global pipeline
        pipeline = _get_pipeline()
        
        # Initialize it if it wasn't already (e.g. fresh server start)
        if not pipeline._indexed:
            pipeline.load_precomputed_index(str(data_dir))

        _tasks[task_id]["status"] = "ranking"

        # 3. Run the ranking pipeline by temporarily overriding the jd_text
        original_jd_text = getattr(pipeline, 'jd_text', None)
        original_parsed_jd = getattr(pipeline, 'parsed_jd', None)
        
        try:
            pipeline.jd_text = jd_text
            pipeline.parsed_jd = parsed
            results = pipeline.rank(top_k=100)
        finally:
            if original_jd_text is not None:
                pipeline.jd_text = original_jd_text
            if original_parsed_jd is not None:
                pipeline.parsed_jd = original_parsed_jd

        _tasks[task_id]["status"] = "exporting"

        # 5. Export to disk — use SEPARATE files so we never overwrite the
        #    main hackathon submission with custom-JD results.
        csv_path = project_root / "output" / "custom_submission.csv"
        json_path = project_root / "output" / "custom_detailed_results.json"
        pipeline.export_submission_csv(results, str(csv_path))
        pipeline.export_detailed_json(results, str(json_path))

        elapsed = time.time() - t0

        # 6. Convert results to frontend format
        frontend_results = _pipeline_results_to_frontend(pipeline, results)

        # 7. Build response
        top_score = frontend_results[0]["final_score"] if frontend_results else 0.0
        pipeline_meta = {
            "candidates_scored": len(frontend_results),
            "retrieval_time_s": round(elapsed, 2),
            "top_fit_score": round(top_score, 4),
            "result_count": len(frontend_results),
            "data_source": "custom_jd_full_pipeline",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        parsed_jd = {
            "role_title": parsed.role_title,
            "seniority": parsed.seniority,
            "min_experience_years": parsed.min_experience_years,
            "must_have_skills": parsed.must_have_skills,
            "nice_to_have_skills": parsed.nice_to_have_skills,
            "domains": parsed.domains,
            "work_mode": parsed.work_mode,
            "intent_facets": parsed.intent_facets,
            "ranking_priorities": parsed.ranking_priorities,
            "responsibilities": parsed.responsibilities,
        }

        _tasks[task_id]["status"] = "done"
        _tasks[task_id]["result"] = {
            "status": "ok",
            "result_count": len(frontend_results),
            "pipeline_meta": pipeline_meta,
            "parsed_jd": parsed_jd,
            "results": frontend_results,
        }
        logger.info(f"Custom JD task {task_id} completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Custom JD task {task_id} failed: {e}")
        _tasks[task_id]["status"] = "error"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["traceback"] = traceback.format_exc()


@app.post("/rank-custom-jd", tags=["Ranking"])
def rank_with_custom_jd(req: RankWithCustomJDRequest):
    """Kick off ranking against a custom JD using the full pipeline.

    Returns a task_id immediately. The pipeline runs in a background
    thread. Poll /task-status/{task_id} for progress and results.

    This runs BM25, FAISS semantic search, cross-encoder reranking,
    feature engineering, XGBoost blending, credibility penalties, and
    contradiction checks — the exact same scoring logic used by /rank
    and /rank-from-files.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    # Verify precomputed indices exist
    if not (data_dir / "bm25_index.pkl").exists():
        raise HTTPException(
            status_code=404,
            detail="Pre-computed BM25 index not found. Run the initial pipeline first.",
        )

    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "status": "queued",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jd_preview": req.jd_text[:120],
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_custom_jd_task,
        args=(task_id, req.jd_text),
        daemon=True,
    )
    thread.start()

    return {
        "status": "accepted",
        "task_id": task_id,
        "message": "Pipeline started. Poll /task-status/{task_id} for progress.",
    }


@app.get("/task-status/{task_id}", tags=["Tasks"])
def get_task_status(task_id: str):
    """Poll for the status and result of a background ranking task."""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    response: dict[str, Any] = {
        "task_id": task_id,
        "status": task["status"],
    }

    if task["status"] == "done":
        response["result"] = task["result"]
    elif task["status"] == "error":
        response["error"] = task["error"]

    return response


from typing import List as _List


@app.post("/upload-resumes", tags=["Upload"])
async def upload_resumes(
    files: _List[UploadFile] = File(default=[]),
    drive_url: str = Form(default=""),
    jd_text: str = Form(default=""),
    top_n: int = Form(default=15),
):
    """Upload resumes (PDF/DOCX/JSON/JSONL) or import from a public Drive folder.

    Parses each file into the internal candidate schema, then runs the
    FULL ranking pipeline — BM25, FAISS semantic search, cross-encoder
    reranking, feature engineering, XGBoost blending, credibility
    penalties, and contradiction checks — identical to /rank-from-files.

    Performance note: For small batches (< 20 resumes), this is fast
    because index building and scoring scale with candidate count.
    """
    t0 = time.time()

    # Clamp top_n
    top_n = max(5, min(50, top_n))

    all_candidates: list[dict] = []
    errors: list[str] = []
    upload_counter = 1

    # ── 1. Process uploaded files ─────────────────────────────────────────
    for f in files:
        try:
            file_bytes = await f.read()
            if not file_bytes or len(file_bytes) < 10:
                errors.append(f"{f.filename}: empty or too small")
                continue

            ext = Path(f.filename or "").suffix.lower()
            if ext not in (".pdf", ".docx", ".json", ".jsonl"):
                errors.append(f"{f.filename}: unsupported format ({ext})")
                continue

            cand_id = f"UPLOAD_{upload_counter:03d}"
            parsed_files = parse_resume_file(file_bytes, f.filename or "file", cand_id)

            for cand in parsed_files:
                # Assign sequential IDs if not already set
                if not cand.get("candidate_id") or not cand["candidate_id"].startswith("UPLOAD_"):
                    cand["candidate_id"] = f"UPLOAD_{upload_counter:03d}"
                upload_counter += 1
                all_candidates.append(cand)

        except Exception as e:
            errors.append(f"{f.filename}: parse error — {str(e)[:100]}")
            logger.error(f"Failed to parse upload '{f.filename}': {e}")

    # ── 2. Process Google Drive link ──────────────────────────────────────
    if drive_url and drive_url.strip():
        try:
            drive_files = fetch_drive_folder(drive_url.strip())
            for fname, fbytes in drive_files:
                try:
                    cand_id = f"UPLOAD_{upload_counter:03d}"
                    parsed_files = parse_resume_file(fbytes, fname, cand_id)
                    for cand in parsed_files:
                        if not cand.get("candidate_id") or not cand["candidate_id"].startswith("UPLOAD_"):
                            cand["candidate_id"] = f"UPLOAD_{upload_counter:03d}"
                        upload_counter += 1
                        all_candidates.append(cand)
                except Exception as e:
                    errors.append(f"Drive:{fname}: {str(e)[:80]}")
        except Exception as e:
            errors.append(f"Drive folder error: {str(e)[:120]}")

    if not all_candidates:
        return {
            "status": "error",
            "message": "No candidates could be parsed from the uploaded files.",
            "errors": errors,
            "result_count": 0,
            "results": [],
        }

    # ── 3. Run the FULL ranking pipeline ──────────────────────────────────
    active_jd_text = jd_text.strip() if jd_text.strip() else DEFAULT_JD_TEXT

    pipeline = RankingPipeline(
        jd_text=active_jd_text,
        use_semantic=True,
        use_reranker=True,
    )

    # Load parsed candidates into the pipeline (same as load_candidates_from_jsonl)
    pipeline.load_candidates_from_list(all_candidates)

    # Build retrieval indices (BM25 + FAISS) for uploaded candidates
    pipeline.build_index()

    # Run the full ranking pipeline — BM25 query, semantic search,
    # cross-encoder reranking, feature engineering, XGBoost blending,
    # credibility penalties, contradiction checks — all applied.
    results = pipeline.rank(top_k=min(top_n, len(all_candidates)))

    # ── 4. Convert to frontend format ─────────────────────────────────────
    frontend_results = _pipeline_results_to_frontend(pipeline, results)

    elapsed = time.time() - t0
    top_score = frontend_results[0]["final_score"] if frontend_results else 0.0

    return {
        "status": "ok",
        "result_count": len(frontend_results),
        "total_parsed": len(all_candidates),
        "errors": errors,
        "pipeline_meta": {
            "candidates_scored": len(frontend_results),
            "retrieval_time_s": round(elapsed, 2),
            "top_fit_score": round(top_score, 4),
            "result_count": len(frontend_results),
            "data_source": "uploaded_resumes_full_pipeline",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": frontend_results,
    }


@app.get("/debug/sample-output", tags=["Debug"])
def debug_sample_output():
    """Return sample output from the last ranking run with pipeline metadata."""
    project_root = Path(__file__).resolve().parent.parent
    json_path = project_root / "output" / "detailed_results.json"
    csv_path = project_root / "output" / "TIER3.csv"

    if not json_path.exists():
        return {"status": "no output file found"}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build live pipeline metadata from actual output
    result_count = len(data)
    top_score = data[0]["final_score"] if result_count > 0 else 0.0

    # Count total candidates scored (CSV has all ranked candidates)
    candidates_scored = result_count
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as cf:
                candidates_scored = max(result_count, sum(1 for _ in cf) - 1)  # minus header
        except Exception:
            pass

    # Retrieval time: read from the file modification timestamp delta
    # (approximate — the JSON is written at end of pipeline)
    try:
        mtime = json_path.stat().st_mtime
        ctime = json_path.stat().st_ctime
        retrieval_time = round(mtime - ctime, 1) if mtime > ctime else 0.0
        # Clamp to reasonable range; if file was overwritten in-place, delta ≈ 0
        if retrieval_time <= 0 or retrieval_time > 300:
            retrieval_time = 0.0
    except Exception:
        retrieval_time = 0.0

    pipeline_meta = {
        "candidates_scored": candidates_scored,
        "retrieval_time_s": retrieval_time,
        "top_fit_score": round(top_score, 4),
        "result_count": result_count,
        "data_source": "output/detailed_results.json",
        "generated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(json_path.stat().st_mtime)
        ),
    }

    return {
        "status": "ok",
        "result_count": result_count,
        "pipeline_meta": pipeline_meta,
        "results": data,
    }

# Mount static files at the end to avoid intercepting API routes
project_root = Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=str(project_root), html=True), name="static")

