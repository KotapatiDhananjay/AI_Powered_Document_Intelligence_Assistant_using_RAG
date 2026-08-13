"""
FAISS vector store management.
Handles creating, saving, loading, searching, and deleting
vector indexes with associated metadata.
"""

import json
import faiss
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ChunkMetadata:
    """Metadata stored alongside each vector in the index."""
    chunk_id: str
    text: str
    page_number: int
    document_name: str
    document_id: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk_id: str
    text: str
    page_number: int
    document_name: str
    document_id: int
    score: float
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """
    FAISS-based vector store with metadata mapping.
    Uses IndexFlatIP (inner product) with normalized vectors for cosine similarity.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata_store: list[ChunkMetadata] = []
        self._create_index()

    def _create_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(self.dimension)

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_list: list[ChunkMetadata],
    ):
        """
        Add vectors and their metadata to the store.

        Args:
            vectors: numpy array of shape (n, dimension), dtype float32.
            metadata_list: List of ChunkMetadata corresponding to each vector.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"Vector count ({len(vectors)}) must match metadata count ({len(metadata_list)})"
            )

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Normalize vectors for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.metadata_store.extend(metadata_list)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        document_ids: Optional[list[int]] = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for the most similar vectors.

        Args:
            query_vector: Query vector of shape (1, dimension).
            top_k: Number of results to return.
            document_ids: Optional filter — only return results from these documents.
            score_threshold: Minimum similarity score to include.

        Returns:
            List of SearchResult objects, sorted by score descending.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        faiss.normalize_L2(query_vector)

        # Search for more results if we need to filter by document
        search_k = top_k * 3 if document_ids else top_k

        scores, indices = self.index.search(query_vector, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if score < score_threshold:
                continue

            meta = self.metadata_store[idx]

            # Filter by document_ids if specified
            if document_ids and meta.document_id not in document_ids:
                continue

            results.append(SearchResult(
                chunk_id=meta.chunk_id,
                text=meta.text,
                page_number=meta.page_number,
                document_name=meta.document_name,
                document_id=meta.document_id,
                score=float(score),
                chunk_index=meta.chunk_index,
                metadata=meta.metadata,
            ))

            if len(results) >= top_k:
                break

        return results

    def delete_document(self, document_id: int):
        """
        Remove all vectors for a specific document.
        Rebuilds the index without the deleted document's vectors.
        """
        if not self.metadata_store:
            return

        # Find indices to keep
        keep_indices = [
            i for i, meta in enumerate(self.metadata_store)
            if meta.document_id != document_id
        ]

        if len(keep_indices) == len(self.metadata_store):
            return  # Nothing to delete

        if not keep_indices:
            # All vectors belong to this document — reset
            self._create_index()
            self.metadata_store = []
            return

        # Reconstruct vectors for kept indices
        kept_vectors = np.array([
            self.index.reconstruct(i) for i in keep_indices
        ], dtype=np.float32)

        kept_metadata = [self.metadata_store[i] for i in keep_indices]

        # Rebuild index
        self._create_index()
        self.metadata_store = []
        # Vectors are already normalized from original add
        self.index.add(kept_vectors)
        self.metadata_store = kept_metadata

    def save(self, directory: str, index_name: str = "index"):
        """
        Save the FAISS index and metadata to disk.

        Args:
            directory: Directory to save files in.
            index_name: Base name for the index files.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = dir_path / f"{index_name}.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata as JSON
        metadata_path = dir_path / f"{index_name}_metadata.json"
        metadata_dicts = [asdict(m) for m in self.metadata_store]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dicts, f, ensure_ascii=False, indent=2)

    def load(self, directory: str, index_name: str = "index") -> bool:
        """
        Load a FAISS index and metadata from disk.

        Args:
            directory: Directory containing the index files.
            index_name: Base name for the index files.

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        dir_path = Path(directory)
        index_path = dir_path / f"{index_name}.faiss"
        metadata_path = dir_path / f"{index_name}_metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dicts = json.load(f)

        self.metadata_store = [
            ChunkMetadata(**m) for m in metadata_dicts
        ]

        return True

    @property
    def total_vectors(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal if self.index else 0

    def get_all_texts(self) -> list[str]:
        """Get all stored chunk texts (used for BM25 keyword search)."""
        return [m.text for m in self.metadata_store]

    def get_metadata_by_index(self, index: int) -> Optional[ChunkMetadata]:
        """Get metadata for a specific vector index."""
        if 0 <= index < len(self.metadata_store):
            return self.metadata_store[index]
        return None
