"""
Utility helpers.
File validation, sanitization, and timing utilities.
"""

import re
import uuid
import time
from pathlib import Path


from fastapi import UploadFile, HTTPException, status

from backend.config import get_settings


def validate_upload_file(file: UploadFile) -> tuple[str, str]:
    """
    Validate an uploaded file for type and size.

    Args:
        file: The uploaded file.

    Returns:
        Tuple of (sanitized_filename, file_extension).

    Raises:
        HTTPException: If validation fails.
    """
    settings = get_settings()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(settings.allowed_extensions)}",
        )

    # Check file size (if content_type header is available)
    if file.size and file.size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    # Sanitize filename
    safe_name = sanitize_filename(file.filename)

    return safe_name, ext


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other attacks.
    Generates a UUID-prefixed name to avoid collisions.
    """
    # Remove path components
    basename = Path(filename).name

    # Remove potentially dangerous characters
    safe = re.sub(r'[^\w\-_. ]', '', basename)

    # Ensure it's not empty
    if not safe or safe.startswith('.'):
        safe = "document"

    # Add UUID prefix for uniqueness
    ext = Path(safe).suffix
    name = Path(safe).stem
    unique_name = f"{uuid.uuid4().hex[:12]}_{name}{ext}"

    return unique_name


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start

    def __str__(self):
        return f"{self.name}: {self.elapsed:.3f}s"
