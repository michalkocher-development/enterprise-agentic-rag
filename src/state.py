"""Definicja stanu grafu (GraphState) przepływającego przez węzły LangGraph."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """Stan reprezentujący cykl życia zapytania w grafie Self-Corrective RAG z pamięcią konwersacyjną."""

    messages: Annotated[List[BaseMessage], add_messages]
    """Historia wiadomości w ramach wątku (thread_id) do obsługi dialogu wieloturowego."""

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

    route: Optional[str]
    """Kierunek wyznaczony przez Router: 'retrieve' (wymaga RAG) lub 'direct_answer' (odpowiedź z pamięci)."""

    graded_verdicts: Optional[List[Dict[str, Any]]]
    """Szczegółowe oceny i uzasadnienia odrzuceń/akceptacji dla każdego chunku z LLM Grader."""

    rewrite_info: Optional[Dict[str, Any]]
    """Metadane autokorekty zapytania (pierwotne, nieudane, nowe zapytanie)."""

