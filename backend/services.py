"""Combined backend services."""

from backend.config import get_settings
from backend.database.database import get_db
from backend.database.models import Conversation, Message
from backend.database.models import Document, Chunk
from backend.database.models import User
from backend.rag.embeddings import get_embedding_model
from backend.rag.generator import (
    generate_answer,
    generate_answer_stream,
    rewrite_query,
    generate_student_content,
)
from backend.rag.loader import load_document
from backend.rag.retriever import HybridRetriever
from backend.rag.splitter import split_document
from backend.rag.vector_store import VectorStore, ChunkMetadata
from backend.schemas import SourceCitation
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncIterator, Optional
import time

# === auth_service.py ===




# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data (should include "sub" with user ID).
        expires_delta: Optional custom expiry duration.

    Returns:
        Encoded JWT string.
    """
    settings = get_settings()
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expiry_minutes)
    )
    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.

    Args:
        token: The JWT string.

    Returns:
        The decoded payload.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency to extract and validate the current user from JWT.

    Usage in routes:
        @router.get("/protected")
        async def protected(user: User = Depends(get_current_user)):
            ...
    """
    payload = decode_token(credentials.credentials)

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
        )

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user

# === document_service.py ===





# Per-user vector store cache
_vector_stores: dict[int, VectorStore] = {}


def get_user_vector_store(user_id: int) -> VectorStore:
    """Get or create the vector store for a specific user."""
    if user_id not in _vector_stores:
        settings = get_settings()
        embedding_model = get_embedding_model(settings.embedding_model)
        store = VectorStore(dimension=embedding_model.dimension)

        # Try to load existing index from disk
        index_dir = str(settings.vector_store_path / str(user_id))
        store.load(index_dir)

        _vector_stores[user_id] = store

    return _vector_stores[user_id]


def save_user_vector_store(user_id: int):
    """Persist the user's vector store to disk."""
    if user_id in _vector_stores:
        settings = get_settings()
        index_dir = str(settings.vector_store_path / str(user_id))
        _vector_stores[user_id].save(index_dir)


async def process_document(
    db: AsyncSession,
    document_id: int,
    user_id: int,
    file_path: str,
):
    """
    Process an uploaded document through the full RAG pipeline.

    Pipeline: Extract text → Chunk → Generate embeddings → Add to vector store

    Args:
        db: Database session.
        document_id: The document's database ID.
        user_id: The owner's user ID.
        file_path: Path to the uploaded file.
    """
    settings = get_settings()
    doc = await db.get(Document, document_id)

    if doc is None:
        return

    try:
        # Step 1: Extract text from document
        doc.status = "extracting"
        await db.commit()

        pages = load_document(file_path, doc.original_filename)

        if not pages:
            doc.status = "error"
            doc.error_message = "No text could be extracted from this document."
            await db.commit()
            return

        doc.total_pages = max(p.page_number for p in pages)

        # Step 2: Chunk the text
        doc.status = "chunking"
        await db.commit()

        chunks = split_document(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            doc.status = "error"
            doc.error_message = "Document could not be split into chunks."
            await db.commit()
            return

        doc.total_chunks = len(chunks)

        # Step 3: Save chunks to database
        db_chunks = []
        for chunk in chunks:
            db_chunk = Chunk(
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                char_count=chunk.metadata.get("char_count", len(chunk.text)),
                word_count=chunk.metadata.get("word_count", len(chunk.text.split())),
                metadata_json=chunk.metadata,
            )
            db_chunks.append(db_chunk)
            db.add(db_chunk)

        await db.flush()

        # Step 4: Generate embeddings
        doc.status = "embedding"
        await db.commit()

        embedding_model = get_embedding_model(settings.embedding_model)
        texts = [chunk.text for chunk in chunks]
        vectors = embedding_model.encode(texts, show_progress=True)

        # Step 5: Add to vector store
        doc.status = "indexing"
        await db.commit()

        vector_store = get_user_vector_store(user_id)

        metadata_list = [
            ChunkMetadata(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                page_number=chunk.page_number,
                document_name=chunk.document_name,
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        vector_store.add_vectors(vectors, metadata_list)

        # Save vector store to disk
        save_user_vector_store(user_id)

        # Mark as ready
        doc.status = "ready"
        await db.commit()

        print(f"[Document Service] Processed '{doc.original_filename}': "
              f"{doc.total_pages} pages, {doc.total_chunks} chunks")

    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)[:500]
        await db.commit()
        print(f"[Document Service] Error processing '{doc.original_filename}': {e}")
        raise


async def delete_document(
    db: AsyncSession,
    document_id: int,
    user_id: int,
):
    """
    Delete a document and its associated data.
    Removes from database, vector store, and filesystem.
    """
    settings = get_settings()

    doc = await db.get(Document, document_id)
    if doc is None:
        return

    # Remove from vector store
    vector_store = get_user_vector_store(user_id)
    vector_store.delete_document(document_id)
    save_user_vector_store(user_id)

    # Remove uploaded file
    file_path = settings.upload_path / doc.filename
    if file_path.exists():
        file_path.unlink()

    # Remove from database (cascades to chunks)
    await db.delete(doc)
    await db.commit()


async def get_user_documents(
    db: AsyncSession,
    user_id: int,
) -> list[Document]:
    """Get all documents for a user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.upload_date.desc())
    )
    return list(result.scalars().all())


async def get_user_stats(
    db: AsyncSession,
    user_id: int,
) -> dict:
    """Get dashboard statistics for a user."""
    from backend.database.models import Conversation, Message

    # Total documents
    doc_count = await db.execute(
        select(func.count(Document.id)).where(Document.user_id == user_id)
    )
    total_documents = doc_count.scalar() or 0

    # Total conversations
    conv_count = await db.execute(
        select(func.count(Conversation.id)).where(Conversation.user_id == user_id)
    )
    total_conversations = conv_count.scalar() or 0

    # Total questions (user messages)
    msg_count = await db.execute(
        select(func.count(Message.id))
        .join(Conversation)
        .where(Conversation.user_id == user_id, Message.role == "user")
    )
    total_questions = msg_count.scalar() or 0

    # Recent documents
    recent = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.upload_date.desc())
        .limit(5)
    )
    recent_documents = list(recent.scalars().all())

    return {
        "total_documents": total_documents,
        "total_conversations": total_conversations,
        "total_questions": total_questions,
        "recent_documents": recent_documents,
    }

# === chat_service.py ===


async def process_question(
    db: AsyncSession,
    user_id: int,
    question: str,
    conversation_id: Optional[int] = None,
    document_ids: Optional[list[int]] = None,
) -> dict:
    """
    Process a user question through the full RAG pipeline.

    Pipeline: Query rewriting → Hybrid retrieval → LLM generation → Save to DB

    Returns:
        Dict with answer, sources, conversation_id, message_id, and timing.
    """
    settings = get_settings()

    # Get or create conversation
    conversation = await _get_or_create_conversation(db, user_id, conversation_id, question)

    # Get conversation history for context
    history = await _get_conversation_history(db, conversation.id)

    # Query rewriting (resolve pronouns using conversation history)
    rewritten_question = await rewrite_query(question, history) if history else question

    # Retrieve relevant chunks
    retrieval_start = time.time()

    vector_store = get_user_vector_store(user_id)
    retriever = HybridRetriever(
        vector_store=vector_store,
        top_k_retrieval=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
    )

    retrieval_result = retriever.retrieve(
        query=rewritten_question,
        document_ids=document_ids,
    )

    retrieval_time = time.time() - retrieval_start

    # Generate answer
    generation_start = time.time()

    if retrieval_result.context_text:
        answer = await generate_answer(
            question=question,
            context=retrieval_result.context_text,
            conversation_history=history,
        )
    else:
        answer = ("I couldn't find relevant information in the uploaded documents "
                  "to answer this question. Please make sure you've uploaded the "
                  "relevant documents and try rephrasing your question.")

    generation_time = time.time() - generation_start

    # Build source citations
    sources = [
        SourceCitation(
            document_name=src.document_name,
            page_number=src.page_number,
            text_preview=src.text_preview,
            score=round(src.score, 4),
        )
        for src in retrieval_result.sources
    ]

    sources_json = [s.model_dump() for s in sources]

    # Save user message
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=question,
    )
    db.add(user_msg)

    # Save assistant message
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources_json=sources_json,
        retrieval_time=round(retrieval_time, 3),
        generation_time=round(generation_time, 3),
    )
    db.add(assistant_msg)

    # Touch conversation to update timestamp
    conversation.updated_at = func.now()

    await db.commit()
    await db.refresh(assistant_msg)

    return {
        "answer": answer,
        "sources": sources,
        "conversation_id": conversation.id,
        "message_id": assistant_msg.id,
        "retrieval_time": round(retrieval_time, 3),
        "generation_time": round(generation_time, 3),
    }


async def process_question_stream(
    db: AsyncSession,
    user_id: int,
    question: str,
    conversation_id: Optional[int] = None,
    document_ids: Optional[list[int]] = None,
) -> AsyncIterator[dict]:
    """
    Stream a RAG answer token by token via SSE.

    Yields:
        Dicts with "type" and "data" for each SSE event:
        - {"type": "sources", "data": [sources]}
        - {"type": "token", "data": "text chunk"}
        - {"type": "done", "data": {metadata}}
    """
    settings = get_settings()

    conversation = await _get_or_create_conversation(db, user_id, conversation_id, question)
    history = await _get_conversation_history(db, conversation.id)

    rewritten_question = await rewrite_query(question, history) if history else question

    # Retrieve
    retrieval_start = time.time()

    vector_store = get_user_vector_store(user_id)
    retriever = HybridRetriever(
        vector_store=vector_store,
        top_k_retrieval=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
    )

    retrieval_result = retriever.retrieve(
        query=rewritten_question,
        document_ids=document_ids,
    )

    retrieval_time = time.time() - retrieval_start

    # Send sources first
    sources = [
        SourceCitation(
            document_name=src.document_name,
            page_number=src.page_number,
            text_preview=src.text_preview,
            score=round(src.score, 4),
        )
        for src in retrieval_result.sources
    ]

    yield {"type": "sources", "data": [s.model_dump() for s in sources]}

    # Stream answer
    generation_start = time.time()
    full_answer = ""

    if retrieval_result.context_text:
        async for token in generate_answer_stream(
            question=question,
            context=retrieval_result.context_text,
            conversation_history=history,
        ):
            full_answer += token
            yield {"type": "token", "data": token}
    else:
        no_context_msg = ("I couldn't find relevant information in the uploaded documents "
                          "to answer this question.")
        full_answer = no_context_msg
        yield {"type": "token", "data": no_context_msg}

    generation_time = time.time() - generation_start

    # Save to database
    user_msg = Message(conversation_id=conversation.id, role="user", content=question)
    db.add(user_msg)

    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=full_answer,
        sources_json=[s.model_dump() for s in sources],
        retrieval_time=round(retrieval_time, 3),
        generation_time=round(generation_time, 3),
    )
    db.add(assistant_msg)
    
    # Touch conversation to update timestamp
    conversation.updated_at = func.now()
    
    await db.commit()
    await db.refresh(assistant_msg)

    yield {
        "type": "done",
        "data": {
            "conversation_id": conversation.id,
            "message_id": assistant_msg.id,
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
        }
    }


async def process_student_request(
    db: AsyncSession,
    user_id: int,
    mode: str,
    document_ids: list[int],
    question: Optional[str] = None,
    count: int = 10,
) -> str:
    """Process a student-mode request (summarize, MCQs, viva, etc.)."""
    from backend.database.models import Chunk, Document

    # Gather text from specified documents
    context_parts = []

    for doc_id in document_ids:
        doc = await db.get(Document, doc_id)
        if doc is None or doc.user_id != user_id:
            continue

        result = await db.execute(
            select(Chunk)
            .where(Chunk.document_id == doc_id)
            .order_by(Chunk.chunk_index)
        )
        chunks = result.scalars().all()

        doc_text = f"\n--- Document: {doc.original_filename} ---\n"
        doc_text += "\n".join(c.text for c in chunks)
        context_parts.append(doc_text)

    if not context_parts:
        return "No document content found for the specified documents."

    # Limit context to avoid token limits (~15000 chars)
    full_context = "\n\n".join(context_parts)
    if len(full_context) > 15000:
        full_context = full_context[:15000] + "\n\n[Content truncated for processing...]"

    return await generate_student_content(
        mode=mode,
        context=full_context,
        question=question,
        count=count,
    )


async def get_user_conversations(
    db: AsyncSession,
    user_id: int,
) -> list[Conversation]:
    """Get all conversations for a user."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_conversation_with_messages(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
) -> Optional[Conversation]:
    """Get a specific conversation with all its messages."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    conversation = result.scalar_one_or_none()

    if conversation:
        # Eagerly load messages
        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        conversation.messages = list(msg_result.scalars().all())

    return conversation


async def _get_or_create_conversation(
    db: AsyncSession,
    user_id: int,
    conversation_id: Optional[int],
    question: str,
) -> Conversation:
    """Get an existing conversation or create a new one."""
    if conversation_id:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            return conversation

    # Create new conversation
    title = question[:100] + "..." if len(question) > 100 else question
    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    await db.flush()
    return conversation


async def _get_conversation_history(
    db: AsyncSession,
    conversation_id: int,
) -> list[dict]:
    """Get conversation history as a list of role/content dicts."""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()

    return [{"role": m.role, "content": m.content} for m in messages]
