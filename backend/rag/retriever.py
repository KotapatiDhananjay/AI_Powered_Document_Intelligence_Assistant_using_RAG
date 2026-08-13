"""
Retrieval module with hybrid search and reranking.
Combines FAISS semantic search with BM25 keyword search
using Reciprocal Rank Fusion, then reranks results.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from backend.rag.embeddings import get_embedding_model
from backend.rag.vector_store import VectorStore, SearchResult


@dataclass
class RetrievalSource:
    """A source citation for the final answer."""
    document_name: str
    page_number: int
    text_preview: str
    score: float


@dataclass
class RetrievalResult:
    """Complete retrieval output with chunks and citations."""
    chunks: list[SearchResult]
    sources: list[RetrievalSource]
    context_text: str  # Concatenated text of all retrieved chunks


class HybridRetriever:
    """
    Two-stage retriever:
    1. Hybrid search: FAISS semantic + BM25 keyword, fused via RRF
    2. Reranking: Cross-encoder rescoring of top candidates
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5,
        rrf_k: int = 60,
        score_threshold: float = 0.15,
    ):
        self.vector_store = vector_store
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.rrf_k = rrf_k
        self.score_threshold = score_threshold
        self.embedding_model = get_embedding_model()
        self._reranker = None

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                print("[Retriever] Loading reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                print("[Retriever] Reranker loaded.")
            except Exception as e:
                print(f"[Retriever] Reranker not available: {e}. Using score-based ranking.")
                self._reranker = None
        return self._reranker

    def retrieve(
        self,
        query: str,
        document_ids: Optional[list[int]] = None,
        use_reranking: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query using hybrid search.

        Args:
            query: The user's question.
            document_ids: Optional filter to specific documents.
            use_reranking: Whether to apply cross-encoder reranking.

        Returns:
            RetrievalResult with chunks, sources, and concatenated context.
        """
        if self.vector_store.total_vectors == 0:
            return RetrievalResult(chunks=[], sources=[], context_text="")

        # Stage 1: Hybrid search
        semantic_results = self._semantic_search(query, document_ids)
        keyword_results = self._keyword_search(query, document_ids)

        # Fuse results using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results, keyword_results
        )

        # Take top candidates
        top_candidates = fused_results[:self.top_k_retrieval]

        if not top_candidates:
            return RetrievalResult(chunks=[], sources=[], context_text="")

        # Stage 2: Reranking
        if use_reranking and len(top_candidates) > 1:
            reranked = self._rerank(query, top_candidates)
            final_results = reranked[:self.top_k_rerank]
        else:
            final_results = top_candidates[:self.top_k_rerank]

        # Build sources (deduplicated by document + page)
        sources = self._build_sources(final_results)

        # Build context text
        context_text = self._build_context(final_results)

        return RetrievalResult(
            chunks=final_results,
            sources=sources,
            context_text=context_text,
        )

    def _semantic_search(
        self,
        query: str,
        document_ids: Optional[list[int]] = None,
    ) -> list[SearchResult]:
        """FAISS-based semantic search."""
        query_vector = self.embedding_model.encode_query(query)

        return self.vector_store.search(
            query_vector=query_vector,
            top_k=self.top_k_retrieval * 2,  # Get more for fusion
            document_ids=document_ids,
            score_threshold=self.score_threshold,
        )

    def _keyword_search(
        self,
        query: str,
        document_ids: Optional[list[int]] = None,
    ) -> list[SearchResult]:
        """BM25-based keyword search over the chunk corpus."""
        all_texts = self.vector_store.get_all_texts()

        if not all_texts:
            return []

        # Tokenize for BM25
        tokenized_corpus = [text.lower().split() for text in all_texts]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Get top results
        top_indices = np.argsort(scores)[::-1][:self.top_k_retrieval * 2]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            meta = self.vector_store.get_metadata_by_index(int(idx))
            if meta is None:
                continue

            if document_ids and meta.document_id not in document_ids:
                continue

            results.append(SearchResult(
                chunk_id=meta.chunk_id,
                text=meta.text,
                page_number=meta.page_number,
                document_name=meta.document_name,
                document_id=meta.document_id,
                score=float(scores[idx]),
                chunk_index=meta.chunk_index,
                metadata=meta.metadata,
            ))

        return results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[SearchResult],
        keyword_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        RRF score = sum(1 / (k + rank)) for each list where the item appears.
        """
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        # Score from semantic results
        for rank, result in enumerate(semantic_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + \
                1.0 / (self.rrf_k + rank + 1)
            result_map[result.chunk_id] = result

        # Score from keyword results
        for rank, result in enumerate(keyword_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + \
                1.0 / (self.rrf_k + rank + 1)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            # Override score with RRF score
            fused_results.append(SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                page_number=result.page_number,
                document_name=result.document_name,
                document_id=result.document_id,
                score=rrf_scores[chunk_id],
                chunk_index=result.chunk_index,
                metadata=result.metadata,
            ))

        return fused_results

    def _rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Rerank candidates using a cross-encoder model."""
        reranker = self._get_reranker()

        if reranker is None:
            # Fall back to existing scores
            return sorted(candidates, key=lambda r: r.score, reverse=True)

        # Prepare query-document pairs
        pairs = [(query, result.text) for result in candidates]

        # Score with cross-encoder
        scores = reranker.predict(pairs)

        # Attach scores and sort
        scored_results = []
        for result, score in zip(candidates, scores):
            scored_results.append(SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                page_number=result.page_number,
                document_name=result.document_name,
                document_id=result.document_id,
                score=float(score),
                chunk_index=result.chunk_index,
                metadata=result.metadata,
            ))

        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results

    def _build_sources(self, results: list[SearchResult]) -> list[RetrievalSource]:
        """Build deduplicated source citations."""
        seen = set()
        sources = []

        for result in results:
            key = (result.document_name, result.page_number)
            if key not in seen:
                seen.add(key)
                sources.append(RetrievalSource(
                    document_name=result.document_name,
                    page_number=result.page_number,
                    text_preview=result.text[:200] + "..." if len(result.text) > 200 else result.text,
                    score=result.score,
                ))

        return sources

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build the context text to send to the LLM."""
        context_parts = []

        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result.document_name}, Page {result.page_number}]\n"
                f"{result.text}"
            )

        return "\n\n---\n\n".join(context_parts)
