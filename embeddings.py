"""Local embedding model — no API key required.

Default model: BAAI/bge-base-en-v1.5 (768-dim, top MTEB ranking).
Swap via EMBEDDING_MODEL env var.  See settings.py for alternatives.

Asymmetric retrieval:
  - Documents are encoded with get_embedding() / get_embeddings() — no prefix.
  - Queries are encoded with get_query_embedding() — applies QUERY_INSTRUCTION
    prefix if set (recommended for bge, e5, nomic models).
  - For symmetric models (all-mpnet-base-v2, all-MiniLM*) leave
    QUERY_INSTRUCTION empty; both functions behave identically.

The model is loaded lazily on first call and cached as a module-level
singleton (~1-2s warm-up, then near-instant).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from settings import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def get_embedding(text: str) -> np.ndarray:
    """Encode a document/passage (no instruction prefix).

    Use this when indexing knowledge base chunks or storing documents.
    """
    return _get_model().encode(text, convert_to_numpy=True).astype("float32")


def get_query_embedding(text: str) -> np.ndarray:
    """Encode a search query, applying QUERY_INSTRUCTION prefix when set.

    Use this for user queries at retrieval time.  With asymmetric models
    (bge, e5, nomic) the prefix substantially improves recall.
    Example: set QUERY_INSTRUCTION="Represent this sentence for searching relevant passages: "
    """
    if settings.QUERY_INSTRUCTION:
        text = settings.QUERY_INSTRUCTION + text
    return _get_model().encode(text, convert_to_numpy=True).astype("float32")


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Batch encode a list of document texts (no prefix).

    Much faster than calling get_embedding() in a loop.
    Used by vector_store.add_documents() for bulk indexing.
    """
    return _get_model().encode(texts, convert_to_numpy=True).astype("float32")


def get_dim() -> int:
    """Return the actual embedding dimension of the loaded model."""
    return _get_model().get_sentence_embedding_dimension()

