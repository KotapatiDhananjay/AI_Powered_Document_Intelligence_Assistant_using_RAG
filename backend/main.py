"""
AI Document Intelligence Assistant — FastAPI Application
Main entry point with lifespan management, CORS, and route registration.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.database.database import create_tables
from backend.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs on startup: create directories, database tables, load models.
    """
    settings = get_settings()

    print("=" * 60)
    print("  AI Document Intelligence Assistant")
    print("  Starting up...")
    print("=" * 60)

    # Create required directories
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)

    # Create database tables
    try:
        await create_tables()
        print("[Startup] Database tables created/verified")
    except Exception as e:
        print(f"[Startup] Database error: {e}")
        print("[Startup] Make sure your database server is running or the SQLite path is writable")

    # Pre-load embedding model (lazy, will load on first use)
    print(f"[Startup] Embedding model: {settings.embedding_model}")
    print(f"[Startup] LLM provider: {settings.llm_provider}")
    print(f"[Startup] Upload directory: {settings.upload_path}")
    print(f"[Startup] Vector store: {settings.vector_store_path}")

    print("=" * 60)
    print("  Server ready! Open http://localhost:8000")
    print("  API docs: http://localhost:8000/docs")
    print("=" * 60)

    yield

    # Shutdown
    print("[Shutdown] Saving vector stores...")
    from backend.services import _vector_stores, save_user_vector_store
    for user_id in _vector_stores:
        save_user_vector_store(user_id)
    print("[Shutdown] Done.")


# Create FastAPI app
app = FastAPI(
    title="AI Document Intelligence Assistant",
    description=(
        "RAG-based document intelligence platform. "
        "Upload documents, ask questions in natural language, "
        "and get grounded answers with source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(api_router)

# Serve frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path), html=True), name="static")


# --- Global Exception Handlers ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors."""
    print(f"[Error] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again."},
    )


# --- Root Endpoint ---

@app.get("/", tags=["Root"], response_class=RedirectResponse)
async def root():
    """Root endpoint — redirect to the frontend landing page."""
    return RedirectResponse(url="/static/index.html", status_code=302)


@app.get("/health", tags=["Root"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
