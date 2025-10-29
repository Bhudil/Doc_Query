from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    """Represents a single message in chat history"""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")

class QueryRequest(BaseModel):
    """Request model for /query endpoint"""
    question: str = Field(..., description="The question to ask", min_length=1)
    chat_history: Optional[List[ChatMessage]] = Field(
        default=[], 
        description="Previous conversation history"
    )

class Source(BaseModel):
    """Represents a source document"""
    page: int = Field(..., description="Page number in the PDF")
    content: str = Field(..., description="Relevant text content")

class QueryResponse(BaseModel):
    """Response model for /query endpoint"""
    answer: str = Field(..., description="The answer to the question")
    sources: List[Source] = Field(..., description="Source documents used")
    cached: bool = Field(..., description="Whether response was cached")

class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    faiss_loaded: bool
    bm25_loaded: bool
    llm_loaded: bool