# main.py (FastAPI Backend)
import os
import uuid
import json
import shutil
import faiss
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
# LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.google_genai import GoogleGenAI
import logging

# Google GenAI (new SDK)
#from google import genai

#from llama_index.core.settings import Settings
#from llama_index.embeddings.gemini import GeminiEmbedding
#from llama_index.core.schema import TextNode

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------
# FastAPI App Setup : Initialize
# ---------------------------------------------------------------------
app = FastAPI(title="PDF Chatbot with RAG")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
UPLOAD_DIR = "uploads"
SESSIONS_DIR = "sessions"
PERSIST_DIR = "vector_stores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# Google Gemini Client + Embeddings
# ---------------------------------------------------------------------
""" GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # Set your GEMINI API key as environment variable
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in environment variables") """

""" GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   # Set your GOOGLE API key as environment variable
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY not found in environment variables") """

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set your GEMINI API key as environment variable
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

#genai.configure(api_key=GOOGLE_API_KEY)

# ---------------------------------------------------------------------
# In-memory session + chat store (replace with DB in production)
# ---------------------------------------------------------------------
""" sessions = {}
chat_messages = {} """

# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------
""" class SessionCreate(BaseModel):
    name: str

class Session(BaseModel):
    id: str
    name: str
    created_at: datetime """

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

# Global variables (in production, use a proper database)
chat_engines = {}  # session_id -> chat engine
vector_indexes = {}  # session_id -> vector index

#client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize embeddings + set globally in LlamaIndex
#embed_model = GeminiEmbedding(model_name="models/embedding-001", client=client)
#Settings.embed_model = embed_model


# Initialize embedding model and LLM
embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    # embed_batch_size=100, 
    api_key=GEMINI_API_KEY)
llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
""" def get_vector_store(session_id: str):
    Get or create FAISS vector store for a session
    persist_path = os.path.join(PERSIST_DIR, session_id)
    if os.path.exists(persist_path):
        vector_store = FaissVectorStore.from_persist_dir(persist_path)
    else:
        test_embedding = embed_model.get_text_embedding("hello world")
        dimension = len(test_embedding)
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
    return vector_store

def get_index(session_id: str, vector_store):
    Get or create index for a session
    persist_path = os.path.join(PERSIST_DIR, session_id)
    if os.path.exists(persist_path):
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=persist_path
        )
        index = load_index_from_storage(storage_context=storage_context)
    else:
        index = VectorStoreIndex([], vector_store=vector_store)
    return index

def save_index(session_id: str, index):
    Persist FAISS index
    persist_path = os.path.join(PERSIST_DIR, session_id)
    os.makedirs(persist_path, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_path) """

# ---------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------
""" @app.get("/")
async def read_root():
    return {"message": "PDF Chatbot API"} """

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...), session_id: Optional[str] = Form(None)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Create or use existing session
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session_dir = f"uploads/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        # Save uploaded files
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            file_path = os.path.join(session_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        
        # Load documents and create index
        documents = SimpleDirectoryReader(session_dir).load_data()
        
        # Create vector store
        vector_store_dir = f"vector_stores/{session_id}"
        os.makedirs(vector_store_dir, exist_ok=True)
        if hasattr(embed_model, 'embed_dim'):
            dimension = embed_model.embed_dim
        else:
            # Default dimension for fallback models
            dimension = 768
        print(dimension)
        # Create FAISS index
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            embed_model=embed_model
        )
        # index.storage_context.persist()
        # Store index for later use
        vector_indexes[session_id] = index
        print(dimension)
        # Create chat engine with memory
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
        print(dimension)
        chat_engines[session_id] = chat_engine
        
        # Create session record
        session_data = ChatSession(
            id=session_id,
            name=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            created_at=datetime.now().isoformat()
        )
        
        with open(f"sessions/{session_id}.json", "w") as f:
            json.dump(session_data.dict(), f)
        
        return {"session_id": session_id, "message": "Files uploaded successfully"}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if request.session_id not in chat_engines:
            raise HTTPException(status_code=404, detail="Session not found")
        
        chat_engine = chat_engines[request.session_id]
        response = chat_engine.chat(request.message)
        
        # Update session with new message
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
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    try:
        sessions = []
        for filename in os.listdir("sessions"):
            if filename.endswith(".json"):
                with open(f"sessions/{filename}", "r") as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
        
        return sessions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    try:
        session_file = f"sessions/{session_id}.json"
        if not os.path.exists(session_file):
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(session_file, "r") as f:
            session_data = json.load(f)
        
        return session_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        # Remove session file
        session_file = f"sessions/{session_id}.json"
        if os.path.exists(session_file):
            os.remove(session_file)
        
        # Remove uploads
        upload_dir = f"uploads/{session_id}"
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        
        # Remove vector store
        vector_dir = f"vector_stores/{session_id}"
        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir)
        
        # Remove from memory
        if session_id in chat_engines:
            del chat_engines[session_id]
        if session_id in vector_indexes:
            del vector_indexes[session_id]
        
        return {"message": "Session deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


""" # ✅ Upload + Process PDFs into a specific session
@app.post("/api/process-documents")
async def process_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        saved_files = []
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")

            file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file_path)

        documents = SimpleDirectoryReader(input_files=saved_files).load_data()

        vector_store = get_vector_store(session_id)
        index = get_index(session_id, vector_store)

        for doc in documents:
            node = TextNode(text=doc.text)
            index.insert_nodes([node])

        save_index(session_id, index)

        # Cleanup uploaded files
        for file_path in saved_files:
            os.remove(file_path)

        return {"message": f"Processed {len(files)} documents successfully into session {session_id}"}
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def get_sessions():
    return [
        Session(id=s["id"], name=s["name"], created_at=s["created_at"])
        for s in sessions.values()
    ]

@app.post("/api/sessions")
async def create_session(session: SessionCreate):
    session_id = str(uuid.uuid4())
    new_session = {
        "id": session_id,
        "name": session.name,
        "created_at": datetime.now(),
    }
    sessions[session_id] = new_session
    chat_messages[session_id] = []
    return new_session

@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, session: SessionCreate):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["name"] = session.name
    return sessions[session_id]

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    chat_messages.pop(session_id, None)

    persist_path = os.path.join(PERSIST_DIR, session_id)
    if os.path.exists(persist_path):
        import shutil
        shutil.rmtree(persist_path)

    return {"message": "Session deleted successfully"}

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    if session_id not in chat_messages:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_messages[session_id]

@app.delete("/api/sessions/{session_id}/messages")
async def clear_session_messages(session_id: str):
    if session_id not in chat_messages:
        raise HTTPException(status_code=404, detail="Session not found")
    chat_messages[session_id] = []
    return {"message": "Chat messages cleared successfully"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now(),
    }
    chat_messages[request.session_id].append(user_message)

    vector_store = get_vector_store(request.session_id)
    index = get_index(request.session_id, vector_store)

    # If no docs exist yet, fallback to Gemini direct response
    if index.docstore.docs == {}:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=request.message,
        )
        answer = response.candidates[0].content.parts[0].text
    else:
        query_engine = index.as_query_engine(similarity_top_k=5)
        response = query_engine.query(request.message)
        answer = str(response)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now(),
    }
    chat_messages[request.session_id].append(assistant_message)

    return ChatResponse(response=answer)
 """
# ---------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------
""" @app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
 """
 # Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




    