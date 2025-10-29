import streamlit as st
import requests
from typing import List, Dict
import os

st.set_page_config(page_title="Document Q&A Assistant", layout="wide")
st.title("Document Q&A Assistant")

# Backend API URL (can be overridden by environment variable)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def check_backend_health() -> Dict:
    """Check if backend is running and healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def query_backend(question: str, chat_history: List[Dict]) -> Dict:
    """
    Send question to backend API
    
    Args:
        question: The question to ask
        chat_history: Previous conversation history
        
    Returns:
        Response from backend with answer and sources
    """
    try:
        payload = {
            "question": question,
            "chat_history": chat_history
        }
        
        response = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Error: Backend returned status {response.status_code}",
                "sources": [],
                "cached": False
            }
            
    except requests.exceptions.Timeout:
        return {
            "answer": "Error: Request timed out. Please try again.",
            "sources": [],
            "cached": False
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "Error: Cannot connect to backend. Make sure the API server is running.",
            "sources": [],
            "cached": False
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "cached": False
        }

# Sidebar
with st.sidebar:
    st.subheader("System Status")
    
    # Health check
    health = check_backend_health()
    
    if health:
        if health["status"] == "healthy":
            st.success("Backend is running")
            st.info(f"FAISS: {'✓' if health['faiss_loaded'] else '✗'}")
            st.info(f"BM25: {'✓' if health['bm25_loaded'] else '✗'}")
            st.info(f"LLM: {'✓' if health['llm_loaded'] else '✗'}")
        else:
            st.warning("Backend is degraded")
    else:
        st.error("Backend is not running")
        st.info("Start backend with: uvicorn main:app --reload")
    
    st.divider()
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Enter your question"):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            result = query_backend(question, st.session_state.chat_history[:-1])
            
            answer = result["answer"]
            cached = result.get("cached", False)
            
            # Show cache indicator
            if cached:
                st.caption("Retrieved from cache")
            
            st.markdown(answer)
            
            # Show sources in expander
            if result.get("sources"):
                with st.expander("View Sources"):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"**Source {i} (Page {source['page']})**")
                        st.text(source["content"])
                        st.divider()
    
    # Add assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Trim history
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]