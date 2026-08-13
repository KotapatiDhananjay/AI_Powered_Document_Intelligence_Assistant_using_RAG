# AI Powered Document Intelligence Assistant using RAG

An intelligent web application that empowers users to seamlessly interact with their documents using advanced Retrieval-Augmented Generation (RAG). The assistant provides accurate, grounded answers to questions by combining semantic search, keyword search, and cross-encoder reranking. 

## 🌟 Key Features

- **Document Q&A (RAG):** Ask questions and get precise answers derived directly from your uploaded documents, complete with source citations and page numbers.
- **Advanced Hybrid Search:** Combines FAISS-based semantic search with BM25 keyword search using Reciprocal Rank Fusion (RRF) to retrieve the most relevant information.
- **Cross-Encoder Reranking:** Re-ranks the retrieved chunks using a cross-encoder model for optimal accuracy before feeding them to the LLM.
- **Student Toolkit:** specialized generation modes to help students learn:
  - Generate Document Summaries
  - Auto-generate Multiple Choice Questions (MCQs)
  - Create Viva (oral exam) questions and answers
  - Extract key topics and simple explanations
- **User Authentication:** Secure signup, login, and JWT-based session management.

## 🛠️ Technology Stack

- **Backend:** Python, FastAPI, SQLAlchemy (Async), FAISS, Sentence Transformers, rank_bm25
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Database:** SQLite / PostgreSQL (Configurable via SQLAlchemy)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A modern web browser

### 1. Backend Setup

1. Navigate to the project root directory.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your environment variables. Ensure the `.env` file in the root directory contains your necessary LLM API keys and configuration parameters.
5. Start the FastAPI development server:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

### 2. Frontend Setup

The frontend consists of static files (HTML, CSS, JS). You can serve it using any basic web server.
For example, using Python's built-in HTTP server:

1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Start the static file server (e.g., on port 3000 to avoid conflicting with the backend):
   ```bash
   python -m http.server 3000
   ```
3. Open your browser and navigate to `http://localhost:3000/index.html` (or `http://localhost:3000/dashboard.html`).

## 📁 Project Structure

```text
├── backend/            # FastAPI application, database models, schemas, and API routes
│   ├── rag/            # RAG implementation: retriever, vector store, embeddings, and splitter
│   ├── database/       # SQLAlchemy models and database configuration
│   ├── utils/          # Helper utilities
│   └── main.py         # FastAPI application entry point
├── frontend/           # Vanilla JS, HTML, and CSS for the web interface
├── data/               # Default directory for uploaded files and databases
├── vector_store/       # Persistent storage for FAISS indices and metadata
├── requirements.txt    # Python package dependencies
└── .env                # Environment variables (API keys, settings)
```

## 🧠 How it Works

1. **Document Upload:** Documents are parsed and split into overlapping chunks.
2. **Embedding:** Text chunks are passed through a Sentence Transformer model (e.g., `all-MiniLM-L6-v2`) to generate vector embeddings.
3. **Indexing:** Vectors are stored in a FAISS index, while raw text is mapped for BM25 keyword search.
4. **Retrieval:** When a question is asked, the system performs a hybrid search (Semantic + Keyword), fuses the results, and reranks the top candidates.
5. **Generation:** The reranked context is sent to an LLM alongside the user's query to generate an informed, highly accurate response.
