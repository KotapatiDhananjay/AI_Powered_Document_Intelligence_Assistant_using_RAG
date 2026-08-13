# AI Powered Document Intelligence Assistant using RAG

## Overview

AI Powered Document Intelligence Assistant is an intelligent web application that allows users to interact with their documents using Retrieval-Augmented Generation (RAG).

The system retrieves relevant information from uploaded documents using semantic search and keyword search, applies cross-encoder reranking, and provides grounded responses generated from the retrieved document context.

## Key Features

### Document Question Answering

Users can upload documents and ask questions about their content. The system retrieves relevant document sections and generates answers based on the available information.

Responses include source references and page numbers where applicable.

### Hybrid Search

The application combines two retrieval approaches:

* Semantic search using FAISS and Sentence Transformers
* Keyword search using BM25

The results are combined using Reciprocal Rank Fusion (RRF) to improve retrieval quality.

### Cross-Encoder Reranking

Retrieved document chunks are reranked using a cross-encoder model. This helps prioritize the most relevant context before it is passed to the language model.

### Student Toolkit

The application provides additional learning features for students:

* Document summaries
* Multiple Choice Questions (MCQs)
* Viva questions and answers
* Key topic extraction
* Simple explanations of important concepts

### User Authentication

The application provides user authentication using:

* User registration
* Login
* JWT-based session management

## Technology Stack

| Component      | Technology              |
| -------------- | ----------------------- |
| Backend        | Python, FastAPI         |
| Database       | SQLite / PostgreSQL     |
| ORM            | SQLAlchemy Async        |
| Vector Search  | FAISS                   |
| Embeddings     | Sentence Transformers   |
| Keyword Search | BM25                    |
| Frontend       | HTML5, CSS3, JavaScript |
| Authentication | JWT                     |

## RAG Architecture

The application follows a multi-stage Retrieval-Augmented Generation pipeline:

```text
Document Upload
       |
       v
Document Parsing
       |
       v
Text Chunking
       |
       v
Generate Embeddings
       |
       +----------------------+
       |                      |
       v                      v
FAISS Semantic Search    BM25 Keyword Search
       |                      |
       +----------+-----------+
                  |
                  v
       Reciprocal Rank Fusion
                  |
                  v
        Cross-Encoder Reranking
                  |
                  v
        Relevant Document Context
                  |
                  v
              LLM
                  |
                  v
          Grounded Answer
```

## How It Works

### 1. Document Upload

Users upload documents through the web interface. The application extracts the document text and divides it into smaller overlapping chunks.

### 2. Embedding Generation

Each text chunk is converted into a numerical vector using a Sentence Transformer model such as `all-MiniLM-L6-v2`.

### 3. Document Indexing

The generated embeddings are stored in a FAISS vector index.

The original text and metadata are also maintained for keyword-based retrieval using BM25.

### 4. Hybrid Retrieval

When a user submits a question, the system performs:

* Semantic similarity search using FAISS
* Keyword-based search using BM25

The results from both methods are combined using Reciprocal Rank Fusion.

### 5. Cross-Encoder Reranking

The retrieved chunks are passed through a cross-encoder model to determine which pieces of information are most relevant to the user's question.

### 6. Answer Generation

The highest-ranked document chunks are provided as context to the language model along with the user's question.

The model generates an answer based on the retrieved information.

## Getting Started

### Prerequisites

* Python 3.9 or higher
* A modern web browser
* Required LLM API credentials
* Git

## Backend Setup

### 1. Clone the Repository

```bash
git clone https://github.com/KotapatiDhananjay/AI-Powered-Document-Intelligence-Assistant-using-RAG.git
```

Navigate to the project directory:

```bash
cd AI-Powered-Document-Intelligence-Assistant-using-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create or update the `.env` file in the project root and provide the required API keys and configuration values.

Example:

```text
LLM_API_KEY=your_api_key
```

Do not commit API keys or other sensitive credentials to GitHub.

### 5. Start the FastAPI Server

```bash
uvicorn backend.main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

## Frontend Setup

The frontend consists of static HTML, CSS, and JavaScript files.

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Start a local web server:

```bash
python -m http.server 3000
```

Open the application in your browser:

```text
http://localhost:3000/index.html
```

or:

```text
http://localhost:3000/dashboard.html
```

## Project Structure

```text
AI-Powered-Document-Intelligence-Assistant/
│
├── backend/
│   ├── rag/
│   │   ├── retriever/
│   │   ├── vector_store/
│   │   ├── embeddings/
│   │   └── splitter/
│   │
│   ├── database/
│   │   ├── models/
│   │   └── database configuration
│   │
│   ├── utils/
│   └── main.py
│
├── frontend/
│   ├── HTML files
│   ├── CSS files
│   └── JavaScript files
│
├── data/
│   └── Uploaded documents and database files
│
├── vector_store/
│   └── FAISS indexes and metadata
│
├── requirements.txt
├── .env
└── README.md
```

## Student Learning Tools

The system is designed not only for document question answering but also for academic learning.

### Document Summary

Generates a concise summary of the uploaded document.

### Multiple Choice Questions

Automatically generates MCQs based on the document content.

### Viva Preparation

Creates potential viva questions and answers based on the uploaded material.

### Key Topics

Identifies important topics and concepts from the document.

### Simple Explanations

Provides simplified explanations of complex topics to support learning.

## Advantages

* Combines semantic and keyword-based retrieval.
* Uses reranking to improve context relevance.
* Provides document-grounded answers.
* Supports source and page references.
* Provides multiple academic learning tools.
* Supports persistent vector storage.
* Provides authenticated user access.
* Can work with SQLite or PostgreSQL.

## Future Enhancements

* Support for additional document formats.
* Multi-document conversational search.
* Conversation history.
* Improved citation verification.
* Streaming responses.
* Role-based user access.
* Cloud-based vector database support.
* Advanced document visualization.
* Support for additional embedding and LLM models.

## Security Considerations

* Keep API keys in environment variables.
* Never commit `.env` files containing secrets.
* Use secure JWT configuration in production.
* Configure appropriate CORS policies before deployment.
* Use HTTPS when deploying the application publicly.

## License

This project is developed for educational and portfolio purposes.

## Author

Kotapati Dhananjay

GitHub: https://github.com/KotapatiDhananjay
