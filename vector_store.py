"""Document vector store — pgvector (persistent) with in-memory fallback.

Single vector layer: PostgreSQL + pgvector.
When the database is unavailable, falls back to numpy cosine similarity
over an in-memory list.  No FAISS dependency.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────
_db_available = False
_fallback_docs: list[str] = []
_fallback_embeddings: list[np.ndarray] = []


def init(db_enabled: bool) -> None:
    """Called from main.py startup to set the storage backend."""
    global _db_available
    _db_available = db_enabled
    if _db_available:
        logger.info("Vector store: pgvector (persistent)")
    else:
        logger.info("Vector store: in-memory fallback (non-persistent)")


def add_documents(text_chunks: list[str], source: str = "default") -> int:
    """Index document chunks.  Returns the number of chunks stored."""
    if not text_chunks:
        return 0

    if _db_available:
        import query_db
        query_db.store_document_chunks(text_chunks, source=source)
    else:
        from embeddings import get_embeddings
        embeddings = get_embeddings(text_chunks)
        for chunk, emb in zip(text_chunks, embeddings):
            _fallback_docs.append(chunk)
            _fallback_embeddings.append(emb)

    logger.info(f"Indexed {len(text_chunks)} chunks (source={source})")
    return len(text_chunks)


def search(query: str, k: Optional[int] = None) -> list[str]:
    """Semantic search over indexed documents.  Returns chunk texts."""
    if k is None:
        from settings import settings
        k = settings.RETRIEVAL_K

    if _db_available:
        import query_db
        from embeddings import get_query_embedding
        embedding = get_query_embedding(query)
        return query_db.search_document_chunks(embedding, k=k)
    else:
        from embeddings import get_query_embedding
        if not _fallback_docs:
            return []
        qe = get_query_embedding(query)
        sims = []
        for emb in _fallback_embeddings:
            dot = float(np.dot(qe, emb))
            norm = float(np.linalg.norm(qe) * np.linalg.norm(emb))
            sims.append(dot / norm if norm > 0 else 0.0)
        topk = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [_fallback_docs[i] for i in topk]


def has_documents() -> bool:
    """Check if any documents are indexed."""
    if _db_available:
        import query_db
        return query_db.count_document_chunks() > 0
    return len(_fallback_docs) > 0


def count() -> int:
    """Return the number of indexed document chunks."""
    if _db_available:
        import query_db
        return query_db.count_document_chunks()
    return len(_fallback_docs)


def clear(source: Optional[str] = None) -> None:
    """Remove indexed documents (optionally by source)."""
    if _db_available:
        import query_db
        query_db.clear_document_chunks(source=source)
    else:
        _fallback_docs.clear()
        _fallback_embeddings.clear()
    logger.info(f"Cleared document index{f' (source={source})' if source else ''}")
