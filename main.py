from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

from rag import query_rag

load_dotenv()

app = FastAPI(title="OSFI Navigator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    model: str = "gpt-4o-mini"
    domain: Optional[str] = "all"

class Source(BaseModel):
    title: str
    section: str
    excerpt: str
    relevance: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    model_used: str
    grounded: bool
    confidence: float

@app.get("/")
def root():
    return {"status": "OSFI Navigator API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if request.model not in ["gpt-4o-mini", "gpt-4o"]:
        raise HTTPException(status_code=400, detail="Model must be gpt-4o-mini or gpt-4o")
    try:
        result = await query_rag(question=request.question, model=request.model, domain=request.domain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/domains")
def get_domains():
    return {"domains": [
        {"id": "all", "label": "All Domains"},
        {"id": "osfi_e23", "label": "OSFI E-23 — Model Risk Management"},
        {"id": "b20", "label": "OSFI B-20 — Mortgage Underwriting"},
        {"id": "fintrac", "label": "FINTRAC / PCMLTFA — AML/KYC"},
        {"id": "ifrs9", "label": "IFRS 9 — Expected Credit Loss"},
        {"id": "basel3", "label": "Basel III / CAR — Capital Adequacy"},
        {"id": "pipeda", "label": "PIPEDA / Law 25 — Privacy"},
        {"id": "casl", "label": "CASL — Anti-Spam"},
    ]}
