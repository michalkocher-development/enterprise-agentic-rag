"""Definicja stanu grafu (GraphState) przepływającego przez węzły LangGraph."""

from typing import List, Optional, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """Stan reprezentujący cykl życia zapytania w grafie Self-Corrective RAG."""

    question: str
    """Aktualne pytanie (może być modyfikowane przez węzeł rewrite)."""

    original_question: str
    """Niezmienione, pierwotne pytanie zadane przez użytkownika."""

    documents: List[Document]
    """Lista aktualnie przetwarzanych dokumentów/chunków."""

    rerank_scores: List[float]
    """Oceny dopasowania obliczone lokalnie przez Cross-Encoder na GPU RTX 2050."""

    generation: Optional[str]
    """Wygenerowana odpowiedź końcowa (synteza LLM)."""

    retry_count: int
    """Liczba wykonanych pętli samonaprawczych (zapobiega nieskończonym cyklom)."""

    regeneration_count: int
    """Liczba wykonanych prób ponownej generacji po wykryciu halucynacji (max 1)."""

    hallucination_grade: Optional[str]
    """Wynik oceny ugruntowania odpowiedzi ('grounded' / 'not grounded')."""

    answer_grade: Optional[str]
    """Wynik oceny przydatności odpowiedzi względem pytania ('useful' / 'not useful')."""

    web_search_needed: bool
    """Flaga sygnalizująca brak odpowiedzi w wewnętrznych raportach."""
