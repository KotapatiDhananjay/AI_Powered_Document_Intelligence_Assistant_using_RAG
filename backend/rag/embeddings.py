"""
Embedding module.
Wraps Sentence Transformers to generate vector embeddings
for text chunks and queries.
"""

import numpy as np
from typing import Optional


class EmbeddingModel:
    """
    Singleton wrapper around a Sentence Transformer model.
    Generates dense vector embeddings for text.
    """

    _instance: Optional["EmbeddingModel"] = None
    _model = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_name = model_name
            cls._instance._model = None
        return cls._instance

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Embeddings] Loading model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            print(f"[Embeddings] Model loaded. Dimension: {self.dimension}")

    @property
    def dimension(self) -> int:
        """Get the embedding dimension (384 for all-MiniLM-L6-v2)."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into vector embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts to process at once.
            show_progress: Whether to show a progress bar.
            normalize: Whether to L2-normalize embeddings (for cosine similarity).

        Returns:
            numpy array of shape (len(texts), dimension) with dtype float32.
        """
        self._load_model()

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query into a vector embedding.

        Args:
            query: The query text.
            normalize: Whether to L2-normalize.

        Returns:
            numpy array of shape (1, dimension) with dtype float32.
        """
        return self.encode([query], normalize=normalize)


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Get or create the singleton embedding model instance."""
    return EmbeddingModel(model_name)
