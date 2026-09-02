"""Testy integracyjne cyklicznego grafu LangGraph (Self-Corrective Agentic RAG)."""

import pytest
from src.graph import app
from src.state import GraphState


def test_graph_structure():
    """Weryfikuje poprawność zarejestrowanych węzłów w grafie LangGraph."""
    graph_nodes = app.get_graph().nodes
    
    expected_nodes = [
        "retrieve",
        "local_rerank",
        "grade_documents",
        "rewrite_query",
        "generate",
        "hallucination_check",
    ]
    for node in expected_nodes:
        assert node in graph_nodes, f"Węzeł {node} nie został zarejestrowany w grafie."


def test_graph_end_to_end_nvidia():
    """Weryfikuje pełne przejście przez graf dla precyzyjnego pytania o NVIDIA."""
    query = "Ile wynosiła marża brutto GAAP spółki NVIDIA w Q3 FY2025?"

    initial_state: GraphState = {
        "question": query,
        "original_question": query,
        "documents": [],
        "rerank_scores": [],
        "generation": None,
        "retry_count": 0,
        "hallucination_grade": None,
        "answer_grade": None,
        "web_search_needed": False,
    }

    result = app.invoke(initial_state)

    generation = result.get("generation", "")
    print(f"\n[Test Graph Result NVIDIA]:\n{generation}")

    # Sprawdzenie czy kluczowa liczba 74,6% została odnaleziona i poprawnie zacytowana
    assert "74,6%" in generation or "74.6%" in generation, (
        f"Odpowiedź powinna zawierać dokładną wartość marży 74,6%. Otrzymano: {generation}"
    )
    assert result.get("hallucination_grade") == "grounded", "Odpowiedź powinna zostać oceniona jako ugruntowana w faktach."


def test_graph_end_to_end_alphabet():
    """Weryfikuje pełne przejście przez graf dla pytania o Capex Alphabet."""
    query = "Ile wyniosły łączne nakłady inwestycyjne Capex spółki Alphabet w całym 2024 roku?"

    initial_state: GraphState = {
        "question": query,
        "original_question": query,
        "documents": [],
        "rerank_scores": [],
        "generation": None,
        "retry_count": 0,
        "hallucination_grade": None,
        "answer_grade": None,
        "web_search_needed": False,
    }

    result = app.invoke(initial_state)

    generation = result.get("generation", "")
    print(f"\n[Test Graph Result Alphabet]:\n{generation}")

    # Sprawdzenie czy kluczowa liczba 51,4 mld USD została odnaleziona
    assert "51,4" in generation or "51.4" in generation, (
        f"Odpowiedź powinna zawierać kwotę 51,4 mld USD. Otrzymano: {generation}"
    )
    assert result.get("hallucination_grade") == "grounded", "Odpowiedź powinna być wolna od halucynacji."
