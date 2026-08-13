"""Pydantic schemas for the application."""

from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional


# --- Auth Schemas ---

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=100, description="Username")
    password: str = Field(..., min_length=6, max_length=128, description="Password")


class LoginRequest(BaseModel):
    """User login request."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="Password")


class UserResponse(BaseModel):
    """User info response."""
    id: int
    email: str
    username: str
    is_active: bool
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# --- Document Schemas ---

class DocumentResponse(BaseModel):
    """Document info response."""
    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    total_pages: int
    total_chunks: int
    status: str
    error_message: Optional[str] = None
    upload_date: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """List of documents response."""
    documents: list[DocumentResponse]
    total: int


class DocumentStatsResponse(BaseModel):
    """Dashboard statistics."""
    total_documents: int
    total_questions: int
    total_conversations: int
    recent_documents: list[DocumentResponse]


# --- Chat Schemas ---

class ChatRequest(BaseModel):
    """Chat question request."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    conversation_id: Optional[int] = Field(None, description="Existing conversation ID for multi-turn")
    document_ids: Optional[list[int]] = Field(None, description="Filter to specific document IDs")
    stream: bool = Field(False, description="Whether to stream the response")


class SourceCitation(BaseModel):
    """A source citation in the answer."""
    document_name: str
    page_number: int
    text_preview: str
    score: float


class ChatResponse(BaseModel):
    """Chat answer response."""
    answer: str
    sources: list[SourceCitation]
    conversation_id: int
    message_id: int
    retrieval_time: float  # seconds
    generation_time: float  # seconds


class MessageResponse(BaseModel):
    """A single message in a conversation."""
    id: int
    role: str
    content: str
    sources: Optional[list[SourceCitation]] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    """Conversation with messages."""
    id: int
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    messages: list[MessageResponse] = []

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    """List of conversations."""
    conversations: list[ConversationResponse]
    total: int


class StudentRequest(BaseModel):
    """Student mode request."""
    mode: str = Field(..., description="One of: summarize, mcqs, viva, explain, topics")
    document_ids: list[int] = Field(..., description="Document IDs to use")
    question: Optional[str] = Field(None, description="Specific question (for explain mode)")
    count: int = Field(10, ge=1, le=50, description="Number of items to generate")
