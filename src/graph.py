"""Kompilacja cyklicznego grafu stanów LangGraph (Self-Corrective & Adaptive Agentic RAG)."""

from typing import Any, Literal, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    generate_node,
    grade_documents_node,
    hallucination_node,
    rerank_node,
    retrieve_node,
    rewrite_node,
    route_question_node,
)
from src.state import GraphState


def route_decision_edge(state: GraphState) -> Literal["retrieve", "direct_answer"]:
    """Krawędź warunkowa Routera: RAG w bazie dokumentów vs odpowiedź z pamięci rozmowy."""
    return state.get("route", "retrieve")


def decide_to_generate(state: GraphState) -> Literal["generate", "rewrite_query"]:
    """Krawędź warunkowa decydująca czy przejść do generowania, czy przepisać zapytanie.

    Jeśli żaden dokument nie przeszedł gradera i nie wyczerpano limitu prób (retry_count < retry_limit),
    graf kierowany jest do węzła rewrite_query.
    """
    web_search_needed = state.get("web_search_needed", False)
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    retry_limit = state.get("retry_limit", 2)

    # Brak wartościowych dokumentów -> próba autokorekty zapytania
    if (web_search_needed or not documents) and retry_count < retry_limit:
        return "rewrite_query"

    return "generate"


def decide_after_generate(state: GraphState) -> Literal["hallucination_check", "end"]:
    """Jeśli odpowiedź pochodzi z pamięci dialogu (direct_answer), kończymy bez badania ugruntowania w dokumentach."""
    if state.get("route") == "direct_answer":
        return "end"
    return "hallucination_check"


def grade_generation_v_documents_and_question(
    state: GraphState,
) -> Literal["useful", "not useful", "not grounded"]:
    """Krawędź warunkowa weryfikująca czy generacja nie zawiera halucynacji i odpowiada na pytanie."""
    hallucination_grade = state.get("hallucination_grade", "grounded")
    answer_grade = state.get("answer_grade", "useful")
    regeneration_count = state.get("regeneration_count", 0)
    retry_count = state.get("retry_count", 0)
    retry_limit = state.get("retry_limit", 2)
    regeneration_limit = state.get("regeneration_limit", 1)

    if hallucination_grade == "grounded":
        if answer_grade == "useful":
            return "useful"
        # Odpowiedź nie odpowiada na pytanie -> przepisanie zapytania jeśli limit nie wyczerpany
        return "not useful" if retry_count < retry_limit else "useful"

    # Wykryto halucynację -> jednorazowa próba ponownej generacji
    return "not grounded" if (regeneration_count < regeneration_limit and retry_count < retry_limit) else "useful"



# Wspólny checkpointer pamięci konwersacyjnej w pamięci RAM
global_memory = MemorySaver()


def create_agentic_rag_graph(checkpointer: Optional[Any] = global_memory):
    """Tworzy i kompiluje pełny cykliczny graf StateGraph z pamięcią konwersacyjną i Routerem."""
    workflow = StateGraph(GraphState)

    # 1. Rejestracja węzłów
    workflow.add_node("router", route_question_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("local_rerank", rerank_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_node)

    # 2. Punkt startowy -> Router
    workflow.add_edge(START, "router")

    # 3. Krawędź warunkowa: Router decyduje (retrieve vs direct_answer)
    workflow.add_conditional_edges(
        "router",
        route_decision_edge,
        {
            "retrieve": "retrieve",
            "direct_answer": "generate",
        },
    )

    # 4. Potok pobierania i rerankingu
    workflow.add_edge("retrieve", "local_rerank")
    workflow.add_edge("local_rerank", "grade_documents")

    # 5. Krawędź warunkowa: generowanie czy przepisanie zapytania
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # 6. Pętla samonaprawcza
    workflow.add_edge("rewrite_query", "retrieve")

    # 7. Po wygenerowaniu: bezpośredni koniec dla direct_answer lub audyt halucynacji dla RAG
    workflow.add_conditional_edges(
        "generate",
        decide_after_generate,
        {
            "hallucination_check": "hallucination_check",
            "end": END,
        },
    )

    # 8. Krawędź warunkowa ewaluacji halucynacji
    workflow.add_conditional_edges(
        "hallucination_check",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not useful": "rewrite_query",
            "not grounded": "generate",
        },
    )

    compiled = workflow.compile(checkpointer=checkpointer)
    return StatefulAgentGraph(compiled)


class StatefulAgentGraph:
    """Wrapper zapewniający wsteczną kompatybilność z wywołaniami bez jawnego thread_id."""

    def __init__(self, compiled_graph):
        self._graph = compiled_graph

    def invoke(self, input, config=None, **kwargs):
        config = self._ensure_thread_id(config)
        return self._graph.invoke(input, config=config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        config = self._ensure_thread_id(config)
        return self._graph.stream(input, config=config, **kwargs)

    async def astream(self, input, config=None, **kwargs):
        config = self._ensure_thread_id(config)
        async for chunk in self._graph.astream(input, config=config, **kwargs):
            yield chunk

    async def astream_events(self, input, config=None, **kwargs):
        config = self._ensure_thread_id(config)
        async for event in self._graph.astream_events(input, config=config, **kwargs):
            yield event

    @staticmethod
    def _ensure_thread_id(config):
        if config is None:
            return {"configurable": {"thread_id": "default-session"}}
        cfg = dict(config)
        if "configurable" not in cfg or "thread_id" not in cfg["configurable"]:
            configurable = dict(cfg.get("configurable", {}))
            configurable.setdefault("thread_id", "default-session")
            cfg["configurable"] = configurable
        return cfg

    def __getattr__(self, name):
        return getattr(self._graph, name)


# Kompilacja domyślnej aplikacji z pamięcią
app = create_agentic_rag_graph()

