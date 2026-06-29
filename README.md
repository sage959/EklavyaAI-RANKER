# EklavyaAI-RANKER
Eklavya AI is a deterministic, multi-stage hybrid retrieval and ranking pipeline designed specifically for the Redrob Hackathon to identify top-tier Senior AI/ML Engineers. 

Our pipeline fuses Lexical (BM25) and Semantic (FAISS) retrieval, cross-encoder reranking, XGBoost scoring, and rigorous proof-of-work contradiction checks to automatically parse resumes and output highly accurate, explainable candidate rankings.

---

## Quick Start: Generate the CSV Submission
To reproduce our ranking results and generate the final `TIER3.csv` submission file, you only need to run a single command. 
     
        python run_pipeline.py --precomputed

# HOW TO LAUNCH THE RANKING STUDIO UI
To use the interactive Ranking Studio frontend, you must first start the backend API server. If you don't start the server, the "Launch Ranking Studio" button on the landing page will not work!

Step 1: Start the Backend Server
Open your terminal in the project folder and run the following command to start the API:
               
               uvicorn api.main:app --port 8000

Step 2: Open the Landing Page
Once the server is running, double-click the index.html file to open the landing page in your web browser.

Step 3: Launch the Studio
On the landing page, click the 

          Launch Ranking Studio 
button to connect to the backend and start ranking candidates!


### Prerequisites
- Python 3.11+
- Ensure you have installed the required dependencies from `requirements.txt`:
  ```bash
  pip install -r requirements.txt

Pipeline Architecture
Our pipeline evaluates candidates in 4 distinct stages to ensure speed, accuracy, and interpretability:

  Hybrid Retrieval: Combines standard BM25 lexical search with a dense FAISS vector index (powered by local SentenceTransformers). Scores are fused using            Reciprocal Rank Fusion.
  Cross-Encoder Reranking: The top candidates from the hybrid pool are deeply scored for contextual relevance against the Job Description using a lightweight        cross-encoder.
  Feature Engineering & XGBoost: We extract 30+ behavioral and technical signals (e.g., Python depth, MLOps production exposure, GitHub activity). An XGBoost        ranker blends these features with strict rule-based contradiction penalties.
  Proof-of-Work Verification: Ensures the candidate doesn't just list "LangChain wrapper" experience, heavily penalizing resumes lacking genuine                     infrastructure/systems deployment exposure.

Interactive Sandbox
Want to test how the ranking dynamically reacts to different Job Descriptions or test toggling the Semantic Search and Reranker?

Try out our live Streamlit UI running on Hugging Face Spaces:
Eklavya AI Sandbox
Sandbox Link: https://huggingface.co/spaces/sage959/eklavya-ai
You can also run the sandbox locally via:

     streamlit run sandbox_demo.py


Determinism Guarantee
To ensure our ranking results are 100% reproducible on any machine:

Multi-threaded BLAS operations are disabled during inference.
NumPy and PyTorch random seeds are strictly pinned (seed=42).
Final ranking ties are always broken deterministically by candidate_id sorting.
