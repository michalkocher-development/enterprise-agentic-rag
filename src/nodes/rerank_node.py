"""Węzeł rerankingu (Rerank Node) — lokalna inferencja PyTorch Cross-Encoder na GPU RTX 2050."""

from typing import Any, Dict
from src.reranker.local_reranker import get_reranker
from src.state import GraphState


def rerank_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu wykonujący precyzyjny reranking fragmentów na karcie NVIDIA GeForce RTX 2050.

    Args:
        state: Aktualny stan grafu.

    Returns:
        Aktualizacja stanu z zawężoną listą najlepszych dokumentów (top-3) i ich ocenami.
    """
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {"documents": [], "rerank_scores": []}

    reranker = get_reranker()
    
    # Wyliczenie precyzyjnych wag w FP16 na GPU (zwracamy top 4 dla bogatego kontekstu)
    ranked_pairs = reranker.rank(query=question, documents=documents, top_k=4)

    top_documents = [doc for doc, score in ranked_pairs]
    scores = [score for doc, score in ranked_pairs]

    return {
        "documents": top_documents,
        "rerank_scores": scores,
    }
