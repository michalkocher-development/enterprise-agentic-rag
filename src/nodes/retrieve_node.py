"""Węzeł pobierania (Retrieve Node) — przeszukiwanie bazy wektorowej."""

from typing import Any, Dict
from src.retriever.vector_store import get_vector_store
from src.state import GraphState


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu pobierający początkowych kandydatów z bazy wektorowej za pomocą embeddingów Gemini.

    Args:
        state: Aktualny stan grafu.

    Returns:
        Aktualizacja stanu z pobraną listą dokumentów.
    """
    question = state["question"]
    vector_store = get_vector_store()
    
    # Pobranie top 10 najbardziej zbliżonych semantycznie fragmentów (optymalne dla zapytań porównawczych)
    documents = vector_store.similarity_search(query=question, k=10)
    
    return {"documents": documents}
