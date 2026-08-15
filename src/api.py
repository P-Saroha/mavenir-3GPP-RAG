"""
src/api.py
-----------
Minimal FastAPI backend for the 3GPP RAG chatbot.

Endpoints:
  GET  /health  — liveness check
  POST /ask     — answer a question using the full RAG pipeline

Usage:
    uvicorn src.api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag import answer_question

app = FastAPI(
    title="3GPP Release 17 RAG API",
    description="Answers questions grounded in TS 23.501, 23.502, 23.503",
    version="1.0.0",
)


# ── schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    supported: bool
    sources: list[dict]


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = answer_question(request.question)
    return AskResponse(
        answer=result["answer"],
        supported=result["supported"],
        sources=result["sources"],
    )
