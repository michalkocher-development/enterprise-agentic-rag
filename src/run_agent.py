import sys
import os
from pathlib import Path
from typing import Dict, Any

# Dodanie katalogu głównego projektu do ścieżki Pythona
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.graph import app


def run_agentic_rag(query: str) -> Dict[str, Any]:
    """Uruchamia pełny potok grafu stanów dla zadanego pytania i wypisuje postęp węzłów."""
    settings = get_settings()
    
    # Upewnienie się, że tracing LangSmith jest włączony w środowisku
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

    initial_state = {
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

    config = {
        "configurable": {"thread_id": "cli-session"},
        "tags": ["self-corrective-rag", "nvidia-rtx-2050", settings.gemini_model],
        "metadata": {
            "model": settings.gemini_model,
            "reranker": settings.reranker_model_name,
            "device": settings.get_resolved_device(),
        },
    }

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 70)
    print(f">> [ZAPYTANIE]: {query}")
    print(f">> [MODEL]: {settings.gemini_model} | [GPU]: RTX 2050 | [LANGSMITH]: {settings.langchain_project}")
    print("=" * 70)

    full_state = dict(initial_state)

    # Strumieniowanie wywołań kolejnych węzłów grafu
    for event in app.stream(initial_state, config=config):
        for node_name, state_update in event.items():
            full_state.update(state_update)
            print(f"\n[*] [WEZEL: {node_name}]")
            
            if node_name == "retrieve":
                docs = state_update.get("documents", [])
                print(f"    Pobrano wstepnie {len(docs)} kandydatow z bazy wektorowej.")
                
            elif node_name == "local_rerank":
                scores = state_update.get("rerank_scores", [])
                docs = state_update.get("documents", [])
                print(f"    RTX 2050 przefiltrowal do {len(docs)} fragmentow. Oceny: {[round(s, 3) for s in scores]}")
                
            elif node_name == "grade_documents":
                docs = state_update.get("documents", [])
                needed = state_update.get("web_search_needed", False)
                status = "Wszystkie odrzucone (wymagane przepisanie)" if needed else f"Zaakceptowano {len(docs)} relewantnych chunkow"
                print(f"    Ocena merytoryczna LLM: {status}")
                
            elif node_name == "rewrite_query":
                new_q = state_update.get("question", "")
                retries = state_update.get("retry_count", 0)
                print(f"    [AUTOKOREKTA #{retries}]: Nowe zoptymalizowane zapytanie: '{new_q}'")
                
            elif node_name == "generate":
                print("    Wygenerowano odpowiedz analityczna.")
                
            elif node_name == "hallucination_check":
                h_grade = state_update.get("hallucination_grade", "")
                a_grade = state_update.get("answer_grade", "")
                print(f"    Weryfikacja: Ugruntowanie (Groundedness): {h_grade} | Celnosc (Relevance): {a_grade}")

    print("\n" + "=" * 70)
    print(">> [KONCOWA ODPOWIEDZ ANALITYCZNA]:")
    print("=" * 70)
    print(full_state.get("generation", "Brak odpowiedzi."))
    print("=" * 70)
    print(">> [LANGSMITH]: Pelny slad tego zapytania jest dostepny na https://smith.langchain.com\n")

    return full_state


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "Ile wynosiły łączne nakłady inwestycyjne Capex spółki Alphabet w 2024 roku i na co zostały przeznaczone?"
    
    run_agentic_rag(user_query)
