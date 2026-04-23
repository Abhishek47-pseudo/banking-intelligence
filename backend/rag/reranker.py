"""
Re-ranker — MMR-based diversity filtering on retrieved documents.
"""

import logging
from typing import List
import numpy as np
from langchain_core.documents import Document
from backend.rag.embeddings import embed_texts

log = logging.getLogger(__name__)


async def mmr_rerank(
    query: str,
    docs: List[Document],
    top_k: int = 5,
    lambda_mult: float = 0.7,
) -> List[Document]:
    """
    Maximal Marginal Relevance re-ranking.
    Balances relevance (query similarity) vs diversity (doc-to-doc similarity).
    lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity
    
    Args:
        query:       Query string
        docs:        Candidate documents (already retrieved)
        top_k:       Number of documents to return
        lambda_mult: Trade-off parameter (0.7 = slight relevance bias)
    """
    if not docs:
        return []
    if len(docs) <= top_k:
        return docs

    try:
        # Embed all texts
        texts = [query] + [doc.page_content for doc in docs]
        embeddings = await embed_texts(texts)
        
        query_emb = np.array(embeddings[0])
        doc_embs = np.array(embeddings[1:])
        
        # Normalize for cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
        
        # Relevance scores
        relevance = doc_norms @ query_norm
        
        selected_indices: List[int] = []
        remaining = list(range(len(docs)))
        
        for _ in range(min(top_k, len(docs))):
            if not remaining:
                break
            if not selected_indices:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: relevance[i])
            else:
                # MMR score: lambda * relevance - (1-lambda) * max_similarity_to_selected
                selected_embs = doc_norms[selected_indices]
                scores = []
                for i in remaining:
                    sim_to_selected = float(np.max(doc_norms[i] @ selected_embs.T))
                    mmr_score = lambda_mult * relevance[i] - (1 - lambda_mult) * sim_to_selected
                    scores.append((i, mmr_score))
                best = max(scores, key=lambda x: x[1])[0]
            
            selected_indices.append(best)
            remaining.remove(best)
        
        return [docs[i] for i in selected_indices]
    
    except Exception as e:
        log.warning(f"MMR reranking failed, returning original order: {e}")
        return docs[:top_k]
