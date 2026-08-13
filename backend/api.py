"""Combined backend API routes."""

from backend.config import get_settings
from backend.database.database import get_db, AsyncSessionLocal
from backend.database.models import User, Document
from backend.schemas import (
    ChatRequest, ChatResponse, SourceCitation,
    ConversationResponse, ConversationListResponse, MessageResponse,
)
from backend.schemas import DocumentResponse, DocumentListResponse, DocumentStatsResponse
from backend.schemas import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from backend.schemas import StudentRequest
from backend.services import (
    process_question,
    process_question_stream,
    get_user_conversations,
    get_conversation_with_messages,
)
from backend.services import get_current_user
from backend.services import hash_password, verify_password, create_access_token
from backend.services import process_student_request
from backend.services import process_document, delete_document, get_user_documents, get_user_stats
from backend.utils.helpers import validate_upload_file
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import json

# === auth.py ===



api_router = APIRouter()


@api_router.post("/api/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user account."""

    # Check if email already exists
    result = await db.execute(select(User).where(User.email == request.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Check if username already exists
    result = await db.execute(select(User).where(User.username == request.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This username is already taken",
        )

    # Create user
    user = User(
        email=request.email,
        username=request.username,
        hashed_password=hash_password(request.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Generate JWT
    token = create_access_token(data={"sub": str(user.id)})

    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@api_router.post("/api/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT."""

    # Find user by email
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Generate JWT
    token = create_access_token(data={"sub": str(user.id)})

    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )

# === documents.py ===







@api_router.post("/api/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document for processing."""
    settings = get_settings()

    # Validate file
    safe_filename, file_ext = validate_upload_file(file)

    # Save file to disk
    upload_path = settings.upload_path / safe_filename
    try:
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            file_size = len(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # Check file size after saving
    if file_size > settings.max_file_size_bytes:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    # Create database record
    doc = Document(
        user_id=user.id,
        filename=safe_filename,
        original_filename=file.filename or "document",
        file_type=file_ext.lstrip("."),
        file_size=file_size,
        status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Process document in background
    async def _bg_process(document_id: int, user_id: int, fpath: str):
        async with AsyncSessionLocal() as session:
            try:
                await process_document(session, document_id, user_id, fpath)
            except Exception as e:
                print(f"[Background] Processing error: {e}")

    background_tasks.add_task(_bg_process, doc.id, user.id, str(upload_path))

    return DocumentResponse.model_validate(doc)


@api_router.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents for the current user."""
    documents = await get_user_documents(db, user.id)

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in documents],
        total=len(documents),
    )


@api_router.get("/api/documents/stats", response_model=DocumentStatsResponse)
async def get_stats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get dashboard statistics."""
    stats = await get_user_stats(db, user.id)

    return DocumentStatsResponse(
        total_documents=stats["total_documents"],
        total_questions=stats["total_questions"],
        total_conversations=stats["total_conversations"],
        recent_documents=[DocumentResponse.model_validate(d) for d in stats["recent_documents"]],
    )


@api_router.get("/api/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific document."""
    doc = await db.get(Document, document_id)

    if doc is None or doc.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentResponse.model_validate(doc)


@api_router.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and all associated data."""
    doc = await db.get(Document, document_id)

    if doc is None or doc.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    await delete_document(db, document_id, user.id)

# === chat.py ===




@api_router.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question about uploaded documents.
    Returns a grounded answer with source citations.
    """
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            _stream_response(db, user.id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    result = await process_question(
        db=db,
        user_id=user.id,
        question=request.question,
        conversation_id=request.conversation_id,
        document_ids=request.document_ids,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=result["conversation_id"],
        message_id=result["message_id"],
        retrieval_time=result["retrieval_time"],
        generation_time=result["generation_time"],
    )


async def _stream_response(db: AsyncSession, user_id: int, request: ChatRequest):
    """Generate SSE events for streaming responses."""
    async for event in process_question_stream(
        db=db,
        user_id=user_id,
        question=request.question,
        conversation_id=request.conversation_id,
        document_ids=request.document_ids,
    ):
        yield f"data: {json.dumps(event)}\n\n"

    yield "data: [DONE]\n\n"


@api_router.get("/api/conversations", response_model=ConversationListResponse)
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for the current user."""
    conversations = await get_user_conversations(db, user.id)

    conv_responses = []
    for conv in conversations:
        conv_responses.append(ConversationResponse(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        ))

    return ConversationListResponse(
        conversations=conv_responses,
        total=len(conv_responses),
    )


@api_router.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific conversation with all messages."""
    conversation = await get_conversation_with_messages(db, conversation_id, user.id)

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = []
    for msg in conversation.messages:
        sources = None
        if msg.sources_json:
            sources = [SourceCitation(**s) for s in msg.sources_json]

        messages.append(MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            sources=sources,
            retrieval_time=msg.retrieval_time,
            generation_time=msg.generation_time,
            created_at=msg.created_at,
        ))

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=messages,
    )

# === student.py ===





@api_router.post("/api/student/generate")
async def generate_student_content(
    request: StudentRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate student-focused content from documents.

    Modes:
    - **summarize**: Structured document summary
    - **mcqs**: Multiple choice questions
    - **viva**: Oral exam questions with expected answers
    - **explain**: Simple explanation of a topic
    - **topics**: Important topics extraction
    """
    valid_modes = {"summarize", "mcqs", "viva", "explain", "topics"}

    if request.mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode: {request.mode}. Valid modes: {', '.join(valid_modes)}",
        )

    if not request.document_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one document ID is required",
        )

    try:
        content = await process_student_request(
            db=db,
            user_id=user.id,
            mode=request.mode,
            document_ids=request.document_ids,
            question=request.question,
            count=request.count,
        )

        return {
            "mode": request.mode,
            "content": content,
            "document_ids": request.document_ids,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate content: {str(e)}",
        )
