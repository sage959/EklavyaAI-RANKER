import json
import os
from pathlib import Path

data_dir = Path("d:/EKLAVYA AI/data/sample")
data_dir.mkdir(parents=True, exist_ok=True)

c1 = {
  "candidateid": "C-001",
  "profile": {
    "name": "Arjun Mehta",
    "current_title": "Senior ML Engineer",
    "current_company": "Flipkart",
    "location": "Bengaluru, IN",
    "total_experience_years": 5,
    "summary": "Experienced ML Engineer specializing in retrieval and search."
  },
  "careerhistory": [
    {
      "company": "Flipkart",
      "title": "Senior ML Engineer",
      "start_date": "2021-01",
      "end_date": None,
      "is_current": True,
      "description": "Led retrieval infra for product search. Built FAISS-based retrieval layer handling 2M+ QPS in production. Implemented candidate-centric RAG pipeline. SentenceTransformers fine-tuned."
    },
    {
      "company": "Myntra",
      "title": "ML Engineer",
      "start_date": "2019-01",
      "end_date": "2021-01",
      "is_current": False,
      "description": "Visual similarity search with FAISS. Product ranking models deployed to live service."
    }
  ],
  "education": [
    {
      "institution": "IIT Bombay",
      "degree": "B.Tech",
      "field": "CS",
      "year": 2019
    }
  ],
  "skills": {
    "primary": ["FAISS", "RAG", "Python", "SentenceTransformers", "FastAPI"],
    "secondary": ["Machine Learning", "Information Retrieval"],
    "all_text": ""
  },
  "redrobsignals": {
    "opentoworkflag": True,
    "lastactivedate": "2023-10-01", # Assume it's a bit older or recent depending on today
    "recruiterresponserate": 92,
    "avgresponsetimehours": 2.1,
    "noticeperioddays": 30,
    "githubactivityscore": 85,
    "interviewcompletionrate": 100,
    "savedbyrecruiters30d": 14
  }
}

c2 = {
  "candidateid": "C-002",
  "profile": {
    "name": "Priya Nair",
    "current_title": "Research Engineer",
    "current_company": "Samsung Research",
    "location": "Pune, IN",
    "total_experience_years": 4,
    "summary": "AI researcher focused on semantic search and ranking."
  },
  "careerhistory": [
    {
      "company": "Samsung Research",
      "title": "Research Engineer",
      "start_date": "2021-01",
      "end_date": None,
      "is_current": True,
      "description": "Bi-encoder and cross-encoder reranking. Published at ECIR 2023. BM25 and dense retrieval hybrid pipeline."
    }
  ],
  "education": [
    {
      "institution": "IIT Madras",
      "degree": "M.Tech",
      "field": "AI",
      "year": 2020
    }
  ],
  "skills": {
    "primary": ["Cross-encoder", "BM25", "PyTorch", "Semantic Search"],
    "secondary": ["HuggingFace", "Python"],
    "all_text": ""
  },
  "redrobsignals": {
    "opentoworkflag": True,
    "lastactivedate": "2023-10-02",
    "recruiterresponserate": 87,
    "avgresponsetimehours": 3.2,
    "noticeperioddays": 45,
    "githubactivityscore": 72,
    "interviewcompletionrate": 90,
    "savedbyrecruiters30d": 11
  }
}

c3 = {
  "candidateid": "C-003",
  "profile": {
    "name": "Neha Sharma",
    "current_title": "Data Analyst",
    "current_company": "Infosys",
    "location": "Hyderabad, IN",
    "total_experience_years": 1.5,
    "summary": "Looking for entry-level AI roles."
  },
  "careerhistory": [
    {
      "company": "Infosys",
      "title": "Data Analyst",
      "start_date": "2022-01",
      "end_date": None,
      "is_current": True,
      "description": "SQL reporting, basic Python scripting. Used OpenAI API wrappers for internal chatbot. No deployment."
    }
  ],
  "education": [
    {
      "institution": "VIT Vellore",
      "degree": "B.E.",
      "field": "CS",
      "year": 2022
    }
  ],
  "skills": {
    "primary": ["Python", "SQL", "OpenAI API"],
    "secondary": ["LangChain wrapper"],
    "all_text": "ChatGPT wrapper"
  },
  "redrobsignals": {
    "opentoworkflag": True,
    "lastactivedate": "2023-09-01",
    "recruiterresponserate": 65,
    "avgresponsetimehours": 9.2,
    "noticeperioddays": 90,
    "githubactivityscore": 42,
    "interviewcompletionrate": 75,
    "savedbyrecruiters30d": 2
  }
}

for c in [c1, c2, c3]:
    with open(data_dir / f"{c['candidateid']}.json", "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)

print("Sample data generated.")
