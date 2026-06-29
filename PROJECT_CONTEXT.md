# Project Context & Handoff Document
## 1. Project Overview and Goals

**Eklavya AI — Ranking Engine** is an intelligent candidate ranking system designed for the Redrob Hackathon. It processes JSONL dumps of candidate profiles and matches them against Job Descriptions (JDs). The system provides a multi-stage NLP pipeline (hybrid retrieval, cross-encoder reranking, and behavioral feature engineering blended with an XGBoost ML model) to output a highly precise, explainable scored ranking.

The project features a **two-part architecture**:
1. **Full Application**: A FastAPI backend running via Uvicorn, serving a vanilla HTML/JS/CSS frontend. This is the main application architecture.
2. **Sandbox Demo**: A standalone Streamlit application (`sandbox_demo.py`) designed specifically to be hosted on Hugging Face Spaces. This acts as a quick evaluation sandbox for hackathon judges to verify the pipeline's execution end-to-end within a 5-minute budget.

## 2. Current Architecture and Folder Structure
- **Backend**: FastAPI server (`api/main.py`) handling HTTP requests.
- **Frontend**: A vanilla HTML/JS/CSS single-page application (`demo-app.html`, `index.html`) served statically by the FastAPI backend.
- **Sandbox UI**: Streamlit application (`sandbox_demo.py` / `streamlit_app.py`) for the Hugging Face sandbox.
- **Ranking Engine Pipeline**: Found in `ranking/`.
  - Parses JD -> Builds Candidates -> Hybrid Retrieval -> Reranking -> Feature Vectors -> XGBoost + Rule Scoring -> Final Explanations.
- **Models**: Pre-trained XGBoost artifacts stored in `models/`.

**Structure:**
- `/api/` - FastAPI routes and entry point (`main.py`).
- `/ranking/` - Core engine modules.
  - `pipeline.py`: Main Orchestrator.
  - `config.py`: Hardcoded weights, thresholds, and keyword signals.
  - `retrieval_lexical.py`, `retrieval_semantic.py`: BM25 and Vector search.
  - `xgb_model.py`, `xgb_features.py`: XGBoost inference and feature extraction.
  - `scorer.py`, `feature_engineering.py`: Rule-based signals and honeypot detection.
- `/models/` - `xgb_ranker.json`.
- `/data/` and `/dataset/` - Data schemas, mock data, and `sample_candidates.jsonl`.
- `/output/` - Generated results (`TIER3.csv`, `detailed_results.json`).
- `demo-app.html` & `index.html` - The main UI application.
- `sandbox_demo.py` - Streamlit sandbox entry point.
- `requirements.txt` & `requirements_sandbox.txt` - Dependencies.

## 3. Technologies, Frameworks, and Dependencies
- **Language**: Python 3.9+ (Backend), Vanilla JavaScript (Frontend)
- **Web Frameworks**: FastAPI, Uvicorn, Streamlit
- **ML & Search**: 
  - `xgboost` (Blended ranking)
  - `scikit-learn` (Required for loading XGBoost regressor wrappers)
  - `sentence-transformers` (Cross-encoder reranking, semantic embeddings)
  - `faiss-cpu` (Vector index for semantic search)
  - `numpy`, `pandas`
- **Data Parsing**: `pydantic`
- **Frontend**: Plain HTML5, CSS3 with modern UI patterns (glassmorphism, CSS variables), Vanilla JS (no React/Vue).

## 4. Key Design Decisions and Why They Were Made
- **Streaming Candidate Processing**: Massive JSONL datasets are streamed rather than loaded fully into memory to keep RAM usage low.
- **Vanilla JS Frontend**: Avoids build steps (Webpack/Vite) to allow rapid prototyping and direct serving from the FastAPI static folder without complexity.
- **Streamlit for Sandbox**: Added a Streamlit app alongside the main FastAPI app to perfectly meet the hackathon's "Sandbox Demo" requirement for Hugging Face Spaces.
- **Graceful Degradation**: If `xgboost`, `scikit-learn`, or `sentence-transformers` fail to load on the cloud environment, the pipeline gracefully disables them and falls back to rule-based or lexical scoring instead of crashing.
- **Honeypot Handling built into Rules**: Instead of special-casing the 80 honeypot candidates, the system organically downranks them via timeline inconsistency (e.g. experience > company age) and skill inflation (expert with 0 years used) logic inside `scorer.py`.

## 5. Features Completed
- Complete ranking backend pipeline (`load -> chunk -> retrieve -> rerank -> score`).
- Output generation complying with strict schema (`candidate_id,rank,score,reasoning`).
- Fully verifiable Hugging Face Streamlit Sandbox Demo.
- XGBoost blended scoring fixed and integrated.
- Dynamic custom Job Description parsing (Must-haves vs Nice-to-haves).
- Career pivot logic allowing non-standard job titles to rank highly if past roles match ML engineering context.
- Honeypot logic verification.

## 6. APIs and Important Data Flows (FastAPI)
- `GET /debug/sample-output`: Returns the latest output from `output/detailed_results.json` along with pipeline stats metadata.
- `POST /index-candidates`: Ingests `candidates.jsonl`.
- `POST /rank`: Runs the retrieval and scoring, generating the final 100 ranked candidates.
- **Data Flow**: `demo-app.html` -> clicks Re-run -> JS calls backend -> Backend runs `ranking/pipeline.py` -> Backend writes JSON/CSV to `/output/` -> Backend returns results array to UI.

## 7. Known Hugging Face Deployment Quirks (Sandbox)
- **XGBoost dependencies**: Hugging Face Spaces require a `packages.txt` containing `libgomp1` for XGBoost to run.
- **scikit-learn requirement**: The XGBoost model was trained using `XGBRegressor`, meaning `scikit-learn` MUST be present in `requirements.txt` to deserialize the model parameters, otherwise it fails silently.
- **File Uploads**: The 381KB `xgb_ranker.json` must be uploaded directly via the HF UI. Attempting to copy/paste the JSON into the HF text editor causes the browser to freeze and truncates the file.

## 8. Critical Files That Should Not Be Rewritten
- `ranking/scorer.py` and `ranking/pipeline.py`: Carefully tuned logic for exact dataset output compliance.
- `models/xgb_ranker.json`: Trained ML weights. Do not modify manually.
- `demo-app.html`: Core structure should be preserved; extend rather than replace.