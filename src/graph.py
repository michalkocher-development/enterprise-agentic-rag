"""Kompilacja cyklicznego grafu stanów LangGraph (Self-Corrective Agentic RAG)."""

from typing import Literal
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    generate_node,
    grade_documents_node,
    hallucination_node,
    rerank_node,
    retrieve_node,
    rewrite_node,
)
from src.state import GraphState


def decide_to_generate(state: GraphState) -> Literal["generate", "rewrite_query"]:
    """Krawędź warunkowa decydująca czy przejść do generowania, czy przepisać zapytanie.

    Jeśli żaden dokument nie przeszedł gradera i nie wyczerpano limitu prób (retry_count < 2),
    graf kierowany jest do węzła rewrite_query.
    """
    web_search_needed = state.get("web_search_needed", False)
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    # Brak wartościowych dokumentów -> próba autokorekty zapytania
    if (web_search_needed or not documents) and retry_count < 2:
        return "rewrite_query"

    return "generate"


def grade_generation_v_documents_and_question(
    state: GraphState,
) -> Literal["useful", "not useful", "not grounded"]:
    """Krawędź warunkowa weryfikująca czy generacja nie zawiera halucynacji i odpowiada na pytanie."""
    hallucination_grade = state.get("hallucination_grade", "grounded")
    answer_grade = state.get("answer_grade", "useful")
    regeneration_count = state.get("regeneration_count", 0)
    retry_count = state.get("retry_count", 0)

    if hallucination_grade == "grounded":
        if answer_grade == "useful":
            return "useful"
        # Odpowiedź nie odpowiada na pytanie -> przepisanie zapytania jeśli limit nie wyczerpany
        return "not useful" if retry_count < 2 else "useful"

    # Wykryto halucynację -> jednorazowa próba ponownej generacji
    return "not grounded" if (regeneration_count < 1 and retry_count < 2) else "useful"


def create_agentic_rag_graph():
    """Tworzy i kompiluje pełny cykliczny graf StateGraph."""
    workflow = StateGraph(GraphState)

    # Rejestracja węzłów
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("local_rerank", rerank_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_node)

    # Krawędzie statyczne
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "local_rerank")
    workflow.add_edge("local_rerank", "grade_documents")

    # Krawędź warunkowa: czy generować, czy poprawić zapytanie?
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # Pętla samonaprawcza: powrót do wyszukiwania z nowym zapytaniem
    workflow.add_edge("rewrite_query", "retrieve")

    # Weryfikacja jakości wygenerowanej odpowiedzi
    workflow.add_edge("generate", "hallucination_check")

    # Krawędź warunkowa ewaluacji halucynacji
    workflow.add_conditional_edges(
        "hallucination_check",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not useful": "rewrite_query",
            "not grounded": "generate",
        },
    )

    return workflow.compile()


# Kompilacja domyślnej aplikacji
app = create_agentic_rag_graph()
