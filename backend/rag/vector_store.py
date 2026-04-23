"""
FAISS Vector Store — three indices: client profiles, intents, product gaps.
Supports save/load and upsert operations.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.rag.embeddings import get_embeddings

log = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

# Index names
INDEX_PROFILES = "client_profiles_index"
INDEX_INTENTS = "intents_index"
INDEX_GAPS = "product_gaps_index"


class BankingVectorStore:
    """Manages three FAISS indices for the banking intelligence platform."""

    def __init__(self):
        self._embeddings = get_embeddings()
        self._indices: Dict[str, Optional[FAISS]] = {
            INDEX_PROFILES: None,
            INDEX_INTENTS: None,
            INDEX_GAPS: None,
        }

    def _index_path(self, name: str) -> str:
        return os.path.join(FAISS_INDEX_PATH, name)

    def load_all(self) -> None:
        """Load all FAISS indices from disk if they exist."""
        for name in self._indices:
            path = self._index_path(name)
            if os.path.exists(path):
                try:
                    self._indices[name] = FAISS.load_local(
                        path, self._embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    log.info(f"Loaded FAISS index: {name}")
                except Exception as e:
                    log.warning(f"Failed to load FAISS index {name}: {e}")

    def save_all(self) -> None:
        """Persist all FAISS indices to disk."""
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        for name, idx in self._indices.items():
            if idx is not None:
                idx.save_local(self._index_path(name))
                log.info(f"Saved FAISS index: {name}")

    def _upsert(self, index_name: str, docs: List[Document]) -> None:
        """Add documents to a FAISS index, creating it if needed."""
        if not docs:
            return
        if self._indices[index_name] is None:
            self._indices[index_name] = FAISS.from_documents(docs, self._embeddings)
        else:
            self._indices[index_name].add_documents(docs)

    def upsert_profile(
        self,
        client_id: str,
        enriched_text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Upsert an enriched client profile into the profiles index."""
        doc = Document(
            page_content=enriched_text,
            metadata={"client_id": client_id, **metadata},
        )
        self._upsert(INDEX_PROFILES, [doc])

    def upsert_intents(
        self,
        client_id: str,
        intents: List[Dict[str, Any]],
    ) -> None:
        """Upsert individual intent objects into the intents index."""
        docs = []
        for intent in intents:
            text = f"{intent.get('type', '')}: {intent.get('value', '')}"
            docs.append(Document(
                page_content=text,
                metadata={"client_id": client_id, "intent": json.dumps(intent)},
            ))
        self._upsert(INDEX_INTENTS, docs)

    def upsert_product_gaps(
        self,
        client_id: str,
        gaps: List[Dict[str, Any]],
    ) -> None:
        """Upsert product gap summaries into the gaps index."""
        if not gaps:
            return
        gap_text = "; ".join(
            f"{g.get('product')} ({g.get('adoption_rate_among_similar', 0)*100:.0f}% adoption)"
            for g in gaps
        )
        doc = Document(
            page_content=gap_text,
            metadata={"client_id": client_id},
        )
        self._upsert(INDEX_GAPS, [doc])

    def get_profile_retriever(self, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.7):
        """Return an MMR retriever on the profiles index."""
        if self._indices[INDEX_PROFILES] is None:
            return None
        return self._indices[INDEX_PROFILES].as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )

    def similarity_search_profiles(
        self,
        query: str,
        k: int = 20,
        filter_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        """Vector search on profiles index, optionally filtered to specific client_ids."""
        idx = self._indices[INDEX_PROFILES]
        if idx is None:
            return []
        # FAISS doesn't natively support metadata filtering; do post-filter
        results = idx.similarity_search(query, k=k * 3)
        if filter_ids:
            results = [r for r in results if r.metadata.get("client_id") in filter_ids]
        return results[:k]


# Global singleton
_store: Optional[BankingVectorStore] = None


def get_vector_store() -> BankingVectorStore:
    global _store
    if _store is None:
        _store = BankingVectorStore()
    return _store
