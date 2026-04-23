"""
Embedding wrapper — provider-agnostic with automatic fallback.

Priority:
  1. OpenAI text-embedding-3-small (1536 dims) — if OPENAI_API_KEY is set
  2. HuggingFace all-MiniLM-L6-v2  (384 dims)  — local, no API key needed

Note: FAISS indices built with one embedding model are NOT compatible with
indices built with the other (different dimensions).  Delete the FAISS index
directory if you switch providers.
"""

import logging
import os
from typing import List, Optional

from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)

_embeddings: Optional[Embeddings] = None


def _has_openai_key() -> bool:
    val = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(val) and not val.startswith("sk-your")


def _build_embeddings() -> Embeddings:
    """Build and return the best available embeddings model."""

    # ── Try OpenAI first ──────────────────────────────────────────────────────
    if _has_openai_key():
        try:
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            )
            log.info("Embeddings provider: OpenAI (text-embedding-3-small, 1536 dims)")
            return emb
        except Exception as exc:
            log.warning(
                f"OpenAI embeddings init failed ({exc}). "
                "Falling back to HuggingFace sentence-transformers."
            )

    # ── Fallback: HuggingFace all-MiniLM-L6-v2 (local, free) ─────────────────
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info(
            "Embeddings provider: HuggingFace sentence-transformers "
            "(all-MiniLM-L6-v2, 384 dims) [fallback]"
        )
        return emb
    except Exception as exc:
        log.error(
            f"HuggingFace embeddings init also failed ({exc}). "
            "Install sentence-transformers: pip install sentence-transformers"
        )
        raise RuntimeError(
            "No embedding provider available. "
            "Set OPENAI_API_KEY or install sentence-transformers."
        ) from exc


def get_embeddings() -> Embeddings:
    """Lazy singleton: returns the shared embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = _build_embeddings()
    return _embeddings


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings asynchronously."""
    emb = get_embeddings()
    return await emb.aembed_documents(texts)


async def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    emb = get_embeddings()
    return await emb.aembed_query(query)
