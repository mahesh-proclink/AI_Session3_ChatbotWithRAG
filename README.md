# PDF Chatbot with RAG

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload documents (.pdf, .txt, .docx, .xlsx) files and interact with them using natural language queries. It uses **FastAPI** for the backend, **FAISS** for vector storage, and **Google Gemini (via LlamaIndex)** for embeddings and LLM responses.

---

## 🚀 Features
- Upload and process PDF, TXT, DOCX, and XLSX files
- Store and retrieve sessions
- Chat with context-aware responses from documents
- Delete old sessions (including uploaded files and vector stores)
- Frontend interface with file upload, chat messages, and session management

---

## 📂 Project Structure
```
project-root/
│
├── main.py            # FastAPI backend with RAG pipeline
├── static/
│   └── index.html     # Frontend web UI
├── uploads/           # Uploaded PDF files (per session)
├── sessions/          # Metadata + chat history (JSON)
├── vector_stores/     # FAISS vector indexes (per session)
└── requirements.txt   # Python dependencies (to be created)
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd project-root
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv env
source env/bin/activate   # On Linux/Mac
env\Scripts\activate      # On Windows
```

### 3. Install Dependencies
Create a `requirements.txt` file with:
```txt
fastapi
uvicorn
faiss-cpu
python-multipart
pydantic
google-generativeai
llama-index
python-dotenv
pydantic
python-docx
openpyxl
docx2txt
```
Then install:
```bash
pip install -r requirements.txt
```

### 4. Set API Key
Export your **Gemini API key**:
```bash
export GEMINI_API_KEY="your_api_key_here"   # Linux/Mac
setx GEMINI_API_KEY "your_api_key_here"     # Windows
```

### 5. Run Backend
```bash
python main.py
```
The server will start at [http://localhost:8000](http://localhost:8000).

---

## 💻 Usage
1. Open the frontend at [http://localhost:8000](http://localhost:8000).
2. Upload one or more PDF files.
3. Start chatting with your documents.
4. Manage multiple sessions from the sidebar.
5. Delete old sessions if no longer needed.

---

## 🛠️ Tech Stack
- **FastAPI** – Backend API
- **FAISS** – Vector similarity search
- **Google Gemini** – Embeddings + LLM
- **LlamaIndex** – RAG pipeline integration
- **Vanilla JS + HTML + CSS** – Frontend

---

## 📌 Notes
- This implementation uses **in-memory + JSON file storage** for sessions. For production, replace with a proper database.
- CORS is set to allow all origins (`*`). Restrict this in production.
- The assistant persona is named **Alexa** in the system prompt.
- Ensure you have LibreOffice or `docx2txt` for `.docx` parsing and `openpyxl` for `.xlsx` support.
- This project is for demonstration purposes and should be hardened for production use.

---

## 📖 Future Improvements
- Add authentication & user accounts
- Support for non-PDF file types
- Improve UI with React or Vue
- Deploy with Docker & cloud hosting


# Retrieval-Augmented Generation (RAG) : Chatbot With RAG

Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of traditional information retrieval systems (such as search and databases) with the capabilities of generative large language models (LLMs). It redirects the LLM to retrieve relevant information from authoritative, pre-determined knowledge sources.   

- It enables you to customize LLMs without fine-tuning, helping you save money and accelerate time to deployment.
- Large language model (LLM) applications, such as chatbots and other natural language processing (NLP) applications, are unlocking powerful benefits across industries. 
- Organizations use LLMs to reduce operational costs, boost employee productivity, and deliver more-personalized customer experiences.
- To connect many components, RAG pipelines combine several AI toolchains for data ingestion, vector databases, LLMs, and more.
- The goal is to create bots that can answer user questions in various contexts by cross-referencing authoritative knowledge sources.

LLMs are limited to their training data and can produce outdated or inaccurate information, whereas RAG adds a retrieval system to provide current, specific knowledge, improving the LLM's accuracy and depth. Known challenges of LLMs include: 

- Presenting false information when it does not have the answer.
- Presenting out-of-date or generic information when the user expects a specific, current response.
- Creating a response from non-authoritative sources.
- Creating inaccurate responses due to terminology confusion, wherein different training sources use the same terminology to talk about different things.


What is RAG ?
● LLMs (Large Language Models) alone rely on pretraining and may "hallucinate"
● RAG supplements the LLM by retrieving relevant information from external sources (e.g., PDFs, databases, websites)

RAG Flow: Query/Question/Prompt ==>> Retriever (Context) ==>> LLM(Large Lanugae Model) ==>> Response

How RAG Works – Step-by-Step:
User Query
→ Embed Query (turn text into vector)
→ Search Vector DB (find similar document chunks)
→ Retrieve Chunks
→ Send [Query + Retrieved Chunks] to LLM
→ LLM Generates Final Answer

Retrieval-Augmented Generation (RAG) is a hybrid architecture that combines external knowledge retrieval with LLM generation to produce more accurate and grounded responses. RAG adds memory to your agents – enabling smarter, context-rich, and more trustworthy responses.

Understanding RAG Workflow:

 1. Embedding Model -  Converts text into high-dimensional vectors for semantic similarity, Used for both indexing and retrieval
-OpenAI: text-embedding-3-small, text-embedding-ada-002
-Open-source: sentence-transformers, instructor-xl, bge-m3

2. Vector DB - Stores embeddings and enables similarity search, Supports Top-K nearest neighbor search
● FAISS: Fast & lightweight, local use
● Chroma: Easy-to-use Python-native DB
● Weaviate, Pinecone, Qdrant: Cloud-hosted scalable options

3. RAG Prompt Template - a simple example
[CONTEXT] {retrieved_docs}
[QUERY] {user_question}
Using the above context, answer the query.

You can fine-tune with:
● Role prompting (e.g., "You are a legal assistant...")
● Few-shot examples
● Tool-enhanced chains

Implementation RAG using LLAMA-INDEX: LLamaIndex Core Concepts:

- Documents: Your source data (PDFs, text files, web pages)
- Nodes: Chunks of documents with metadata
- Indices: Data structures for efficient retrieval
Load Necessary Modules:
pip install llama-index python-dotenv google-generativeai
pip install "llama-index-embeddings-gemini" "llama-index-llms-gemini"
pip install crewai fastapi uvicorn sentence-transformers

