# Document Q&A Assistant

A RAG-based (Retrieval Augmented Generation) document question-answering system with FastAPI backend and Streamlit frontend.

## Note:
Usually API keys are put in environment variable but since it is open source groq llms, I have decided to push the key itself to the docker image so that the application can simply be hit and run without any setup but do note that the api key expires in a few days.

## Features

- PDF document processing and indexing
- Hybrid retrieval (FAISS + BM25)
- Conversation history awareness
- Query caching for faster responses
- REST API backend
- Interactive chat interface
- Fully containerized with Docker

## Architecture

```
┌─────────────────┐         HTTP          ┌──────────────────┐
│   Streamlit     │  ───────────────────► │   FastAPI        │
│   Frontend      │  {question, history}  │   Backend        │
│   (Port 8501)   │  ◄─────────────────── │   (Port 8000)    │
└─────────────────┘  {answer, sources}    └────────┬─────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────┐
                                          │  FAISS Index    │
                                          │  BM25 Ranking   │
                                          │  LLM (Groq)     │
                                          └─────────────────┘
```

## Tech Stack

**Backend:**
- FastAPI
- LangChain
- FAISS (vector store)
- BM25 (lexical ranking)
- Groq LLM API
- Sentence Transformers

**Frontend:**
- Streamlit
- Requests

**Deployment:**
- Docker
- Docker Compose

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- FAISS index in `faiss_index/` directory

### Run with Docker Compose

```bash
# Clone the repository
git clone Doc_Query

# Start both services
docker-compose up --build

# Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Stop the application

```bash
docker-compose down
```

## Local Development Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Docker Hub Images

Pull pre-built images from Docker Hub:

```bash
# Pull backend image
docker pull devopsdemoo/qa-backend:latest

# Pull frontend image
docker pull devopsdemoo/qa-frontend:latest

# Run with pulled images
docker-compose -f docker-compose.prod.yml up
```

## API Endpoints

### Health Check
```http
GET /health
```

### Query Document
```http
POST /query
Content-Type: application/json

{
  "question": "What is this document about?",
  "chat_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

**Response:**
```json
{
  "answer": "The document discusses...",
  "sources": [
    {
      "page": 1,
      "content": "Relevant excerpt..."
    }
  ],
  "cached": false
}
```

## Project Structure

```
document-qa-assistant/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── retrieval.py         # RAG logic
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
├── frontend/
│   ├── app.py              # Streamlit UI
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
├── faiss_index/            # FAISS vector store
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Environment Variables

### Backend
- `FAISS_PATH`: Path to FAISS index (default: `/app/faiss_index`)
- `GROQ_API_KEY`: Groq API key
- `MODEL_NAME`: LLM model name (default: `llama-3.1-8b-instant`)

### Frontend
- `BACKEND_URL`: Backend API URL (default: `http://localhost:8000`)

## Building Custom Images

### Backend
```bash
cd backend
docker build -t qa-backend:latest .
```

### Frontend
```bash
cd frontend
docker build -t qa-frontend:latest .
```
