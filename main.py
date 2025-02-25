from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from rag_system import MedicalRAG

app = FastAPI(
    title="medicos - Medical RAG API",
    description="A Retrieval-Augmented Generation system for medical questions",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag = MedicalRAG()

class QueryRequest(BaseModel):
    question: str
    use_google_fallback: Optional[bool] = True
    top_k: Optional[int] = 5

class SourceInfo(BaseModel):
    title: str
    source: str
    url: str
    snippet: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    context_source: str
    sources: List[SourceInfo]

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a medical question and return an answer with sources"""
    try:
        result = rag.process_medical_query(
            request.question, 
            use_google_fallback=request.use_google_fallback,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "message": "medicos is operational"}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)