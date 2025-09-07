# main.py (Retrieval-Augmented Generation (RAG): FastAPI Backend)
import os
import uuid
import json
import shutil
import faiss
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai

# Import llama_index components for vector storage and LLM integration
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import logging
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------
app = FastAPI(title="PDF Chatbot with RAG")

# Enable CORS for all origins (use more restrictive settings in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories if they don't exist
UPLOAD_DIR = "uploads"
SESSIONS_DIR = "sessions"
PERSIST_DIR = "vector_stores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration - Google Gemini API setup
# ---------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Must be set in environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Initialize embedding model and LLM using Google GenAI
embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    api_key=GEMINI_API_KEY
)
llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY
)

# ---------------------------------------------------------------------
# Pydantic Models - Define request/response schemas
# ---------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime

class ChatSession(BaseModel):
    id: str
    name: str
    created_at: str
    messages: List[Message] = []

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Global variables for in-memory storage (replace with DB in production)
chat_engines = {}   # Maps session_id -> chat engine
vector_indexes = {} # Maps session_id -> vector index

# ---------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------
@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...), session_id: Optional[str] = Form(None)):
    """
    Upload one or more PDF files, create a session (if not provided),
    build a FAISS vector index, and initialize a chat engine for that session.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Create new session if none provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Save files to session-specific directory
        session_dir = f"uploads/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            file_path = os.path.join(session_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

        # Load documents into memory
        documents = SimpleDirectoryReader(session_dir).load_data()

        # Create FAISS vector store
        vector_store_dir = f"vector_stores/{session_id}"
        os.makedirs(vector_store_dir, exist_ok=True)
        dimension = getattr(embed_model, 'embed_dim', 768)  # Use embed_dim if available
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # Build vector index from documents
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model
        )
        vector_indexes[session_id] = index

        # Create chat engine with memory buffer
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            llm=llm,
            system_prompt=(
                "You are a helpful assistant named Alexa that answers questions based on the provided documents. "
                "If you cannot find the answer in the documents, say so. " 
                "Always be polite and helpful. "
            )
        )
        chat_engines[session_id] = chat_engine

        # Save session metadata
        session_data = ChatSession(
            id=session_id,
            name=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            created_at=datetime.now().isoformat()
        )
        with open(f"sessions/{session_id}.json", "w") as f:
            json.dump(session_data.dict(), f)

        return {"session_id": session_id, "message": "Files uploaded successfully"}

    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Handle user chat message, query the chat engine for a response,
    and store the conversation history in the session file.
    """
    try:
        if request.session_id not in chat_engines:
            raise HTTPException(status_code=404, detail="Session not found")

        chat_engine = chat_engines[request.session_id]
        response = chat_engine.chat(request.message)

        # Save conversation to session file
        session_file = f"sessions/{request.session_id}.json"
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_data = json.load(f)

            session_data["messages"].append({
                "role": "user",
                "content": request.message,
                "timestamp": datetime.now().isoformat()
            })
            session_data["messages"].append({
                "role": "assistant",
                "content": str(response),
                "timestamp": datetime.now().isoformat()
            })

            with open(session_file, "w") as f:
                json.dump(session_data, f)

        return {"response": str(response), "session_id": request.session_id}

    except Exception as e:
        logger.error(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """Return all saved chat sessions."""
    try:
        sessions = []
        for filename in os.listdir("sessions"):
            if filename.endswith(".json"):
                with open(f"sessions/{filename}", "r") as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
        return sessions
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve details of a specific chat session."""
    try:
        session_file = f"sessions/{session_id}.json"
        if not os.path.exists(session_file):
            raise HTTPException(status_code=404, detail="Session not found")

        with open(session_file, "r") as f:
            session_data = json.load(f)
        return session_data
    except Exception as e:
        logger.error(f"Error retrieving session: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and all its associated files (uploads, vector store, metadata).
    """
    try:
        # Remove session metadata file
        session_file = f"sessions/{session_id}.json"
        if os.path.exists(session_file):
            os.remove(session_file)

        # Remove uploaded PDFs
        upload_dir = f"uploads/{session_id}"
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)

        # Remove FAISS vector store
        vector_dir = f"vector_stores/{session_id}"
        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir)

        # Clean up in-memory stores
        chat_engines.pop(session_id, None)
        vector_indexes.pop(session_id, None)

        return {"message": "Session deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

# Serve frontend (index.html from "static" directory)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
