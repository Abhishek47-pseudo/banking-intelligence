"""
Hybrid Retriever — structured SQL pre-filter + FAISS vector search.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from backend.rag.vector_store import get_vector_store
from backend.storage import sql_store

log = logging.getLogger(__name__)


async def hybrid_retrieve(
    query: str,
    client_id: Optional[str] = None,
    income_band: Optional[str] = None,
    has_churn_risk: Optional[bool] = None,
    city: Optional[str] = None,
    top_k: int = 5,
    fetch_k: int = 20,
) -> List[Document]:
    """
    Hybrid retrieval pipeline:
      1. Structured SQL pre-filter → candidate client_ids
      2. FAISS vector search within candidates (MMR re-ranking)
      3. Return top_k documents

    Args:
        query:          Natural language query to embed and search
        client_id:      Exclude this client from results (searching for similar clients)
        income_band:    Pre-filter by income band
        has_churn_risk: Pre-filter by churn risk flag
        city:           Pre-filter by city
        top_k:          Final number of results to return
        fetch_k:        Candidates to fetch before MMR re-ranking
    """
    # Step 1 — SQL pre-filter
    try:
        candidates = await sql_store.filter_profiles(
            income_band=income_band,
            has_churn_risk=has_churn_risk,
            city=city,
            exclude_ids=[client_id] if client_id else None,
        )
        candidate_ids = [c.client_id for c in candidates]
        log.info(f"SQL pre-filter returned {len(candidate_ids)} candidates")
    except Exception as e:
        log.warning(f"SQL pre-filter failed, falling back to full search: {e}")
        candidate_ids = None

    # Step 2 — FAISS vector search
    store = get_vector_store()
    docs = store.similarity_search_profiles(
        query=query,
        k=fetch_k,
        filter_ids=candidate_ids if candidate_ids else None,
    )

    if not docs:
        log.warning("FAISS search returned 0 documents")
        return []

    # Step 3 — MMR re-ranking (via FAISS MMR retriever if available)
    # We already did the similarity search; do simple score-based de-dup
    seen_clients: set = set()
    diverse_docs: List[Document] = []
    for doc in docs:
        cid = doc.metadata.get("client_id")
        if cid not in seen_clients:
            seen_clients.add(cid)
            diverse_docs.append(doc)
        if len(diverse_docs) >= top_k:
            break

    log.info(f"Hybrid retrieval returning {len(diverse_docs)} documents")
    return diverse_docs


def assemble_context(docs: List[Document]) -> str:
    """Assemble retrieved documents into a single context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        cid = doc.metadata.get("client_id", "unknown")
        parts.append(f"[Similar Client {i} — {cid}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts) if parts else "No similar client context available."
