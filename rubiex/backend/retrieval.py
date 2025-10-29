from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
import hashlib
import os
from typing import List, Dict, Optional

class RetrievalEngine:
    """Handles document retrieval and answer generation"""
    
    def __init__(self, faiss_path: str, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize the retrieval engine
        
        Args:
            faiss_path: Path to FAISS index directory
            groq_api_key: Groq API key
            model_name: LLM model name
        """
        self.faiss_path = faiss_path
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        
        # Initialize components
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        self.corpus_docs = []
        self.llm = None
        self.query_cache = {}
        
    def load_resources(self):
        """Load all necessary resources at startup"""
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print(f"Loading FAISS index from {self.faiss_path}...")
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_path}")
        
        self.faiss_index = FAISS.load_local(
            self.faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print("Initializing LLM...")
        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=0.1
        )
        
        print("Extracting documents for BM25...")
        # Get all documents from FAISS
        self.corpus_docs = list(self.faiss_index.docstore._dict.values())
        corpus_texts = [doc.page_content for doc in self.corpus_docs]
        
        print("Building BM25 index...")
        self.bm25 = BM25Okapi([t.split() for t in corpus_texts])
        
        print("All resources loaded successfully!")
        
    def hybrid_retrieve(self, question: str, k: int = 8) -> List:
        """
        Retrieve documents using hybrid FAISS + BM25 approach
        
        Args:
            question: The query question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.faiss_index:
            return []
        
        # Get FAISS results (semantic search)
        faiss_docs = self.faiss_index.similarity_search_with_score(question, k=k)
        
        if not self.bm25 or not self.corpus_docs:
            return [doc for doc, _ in faiss_docs[:5]]
        
        # Get BM25 scores (lexical search)
        query_tokens = question.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        
        # Create score dictionary
        doc_scores = {}
        
        # Add BM25 scores
        for doc, score in zip(self.corpus_docs, bm25_scores):
            doc_content = doc.page_content
            doc_scores[doc_content] = {
                "bm25": score / max_bm25,
                "faiss": 0,
                "doc": doc
            }
        
        # Add FAISS scores
        max_faiss = max([score for _, score in faiss_docs]) if faiss_docs else 1
        for doc, score in faiss_docs:
            doc_content = doc.page_content
            if doc_content in doc_scores:
                doc_scores[doc_content]["faiss"] = 1 - (score / max_faiss) if max_faiss > 0 else 0
            else:
                doc_scores[doc_content] = {
                    "bm25": 0,
                    "faiss": 1 - (score / max_faiss),
                    "doc": doc
                }
        
        # Combine scores (40% BM25, 60% FAISS)
        for content in doc_scores:
            doc_scores[content]["combined"] = (
                0.4 * doc_scores[content]["bm25"] + 
                0.6 * doc_scores[content]["faiss"]
            )
        
        # Sort by combined score
        ranked_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["combined"],
            reverse=True
        )
        
        return [item["doc"] for item in ranked_docs[:5]]
    
    def format_chat_history(self, chat_history: List[Dict], max_turns: int = 3) -> str:
        """
        Format chat history for prompt
        
        Args:
            chat_history: List of chat messages
            max_turns: Maximum number of conversation turns to include
            
        Returns:
            Formatted chat history string
        """
        if not chat_history:
            return "No previous conversation."
        
        # Get last few exchanges
        recent_history = chat_history[-(max_turns * 2):]
        formatted = []
        
        for msg in recent_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted.append(f"User: {content}")
            else:
                # Remove page references from history
                clean_content = content.split("\n\nRelevant pages:")[0]
                formatted.append(f"Assistant: {clean_content}")
        
        return "\n".join(formatted)
    
    def get_cache_key(self, question: str) -> str:
        """Generate cache key from question"""
        return hashlib.sha256(question.lower().strip().encode()).hexdigest()
    
    def generate_answer(self, question: str, chat_history: List[Dict]) -> Dict:
        """
        Generate answer for a question
        
        Args:
            question: The question to answer
            chat_history: Previous conversation history
            
        Returns:
            Dictionary with answer, sources, and cached flag
        """
        # Check cache
        cache_key = self.get_cache_key(question)
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result["cached"] = True
            return cached_result
        
        # Retrieve relevant documents
        relevant_docs = self.hybrid_retrieve(question, k=8)
        
        if not relevant_docs:
            return {
                "answer": "No relevant information found in the document.",
                "sources": [],
                "cached": False
            }
        
        # Build context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Format chat history
        history_context = self.format_chat_history(chat_history, max_turns=3)
        
        # Create prompt
        prompt_template = """You are a professional document assistant. Answer questions based on the provided context and conversation history.

Previous conversation:
{chat_history}

Context: {context}

Question: {question}

Answer:"""
        
        prompt = prompt_template.format(
            chat_history=history_context,
            context=context,
            question=question
        )
        
        # Get answer from LLM
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Extract source information
        sources = []
        seen_pages = set()
        for doc in relevant_docs[:3]:
            page = doc.metadata.get('page', 0) + 1
            if page not in seen_pages:
                sources.append({
                    "page": page,
                    "content": doc.page_content[:200] + "..."  # First 200 chars
                })
                seen_pages.add(page)
        
        # Add page references to answer
        pages = sorted([s["page"] for s in sources])
        if pages:
            answer += f"\n\nRelevant pages: {', '.join(map(str, pages))}"
        
        result = {
            "answer": answer,
            "sources": sources,
            "cached": False
        }
        
        # Cache the result
        self.query_cache[cache_key] = result
        
        return result
    
    def is_ready(self) -> Dict[str, bool]:
        """Check if all components are loaded"""
        return {
            "faiss_loaded": self.faiss_index is not None,
            "bm25_loaded": self.bm25 is not None,
            "llm_loaded": self.llm is not None
        }