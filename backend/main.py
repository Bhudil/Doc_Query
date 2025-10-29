from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QueryRequest, QueryResponse, HealthResponse, Source
from retrieval import RetrievalEngine
import os
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="RAG-based document question answering system",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retrieval engine instance
retrieval_engine = None

@app.on_event("startup")
async def startup_event():
    """
    This runs when the FastAPI server starts.
    We load all resources here so they're ready for requests.
    """
    global retrieval_engine
    
    print("=" * 50)
    print("Starting Document Q&A API...")
    print("=" * 50)
    
    # Configuration - use environment variables or defaults
    FAISS_PATH = os.getenv("FAISS_PATH", "../faiss_index")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    
    try:
        # Initialize retrieval engine
        retrieval_engine = RetrievalEngine(
            faiss_path=FAISS_PATH,
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME
        )
        
        # Load all resources
        retrieval_engine.load_resources()
        
        print("=" * 50)
        print("API is ready to accept requests!")
        print("=" * 50)
        
    except Exception as e:
        print(f"ERROR during startup: {str(e)}")
        raise

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - just a welcome message"""
    return {
        "message": "Document Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "API documentation",
            "/health": "Health check",
            "/query": "Ask questions (POST)"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of all components.
    """
    if retrieval_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    status = retrieval_engine.is_ready()
    
    all_ready = all(status.values())
    
    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        faiss_loaded=status["faiss_loaded"],
        bm25_loaded=status["bm25_loaded"],
        llm_loaded=status["llm_loaded"]
    )

@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query_document(request: QueryRequest):
    """
    Main endpoint for asking questions.
    
    Args:
        request: QueryRequest with question and optional chat_history
        
    Returns:
        QueryResponse with answer, sources, and cached flag
    """
    if retrieval_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please wait and try again."
        )
    
    try:
        # Convert Pydantic models to dicts for processing
        chat_history = [msg.dict() for msg in request.chat_history]
        
        # Generate answer
        result = retrieval_engine.generate_answer(
            question=request.question,
            chat_history=chat_history
        )
        
        # Convert sources to Pydantic models
        sources = [Source(**src) for src in result["sources"]]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            cached=result["cached"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

# Run with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)