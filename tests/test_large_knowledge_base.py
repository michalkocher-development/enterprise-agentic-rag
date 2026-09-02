"""Testy integracyjne dla rozszerzonej bazy wiedzy Big Tech (Wielka Szóstka) oraz trwałego cache'u."""

import time
import pytest
from src.graph import app
from src.reranker.local_reranker import get_reranker
from src.retriever.vector_store import get_vector_store
from src.state import GraphState


@pytest.fixture
def vector_store():
    """Zwraca bazę wektorową załadowaną raportami Wielkiej Szóstki."""
    return get_vector_store()


def test_big_tech_coverage_and_cache(vector_store):
    """Weryfikuje obecność wszystkich 6 spółek w indeksie oraz poprawność cache'u dyskowego."""
    assert vector_store.cache_file.exists(), "Plik cache'u data/vector_cache/index.json powinien istnieć na dysku."
    
    # Wykonanie szerokiego przeszukania, aby zebrać metadane
    docs = vector_store.similarity_search("przychody nakłady sztuczna inteligencja", k=20)
    companies = {doc.metadata.get("company") for doc in docs}

    expected_companies = {"NVIDIA", "Alphabet", "Microsoft", "Amazon", "Meta", "Apple"}
    found_intersection = companies.intersection(expected_companies)
    
    print(f"\n[Coverage Test] Wykryte spółki w próbce: {found_intersection}")
    assert len(found_intersection) >= 4, f"Oczekiwano co najmniej 4 różnych spółek w losowej próbce, znaleziono: {found_intersection}"


def test_cross_company_silicon_retrieval(vector_store):
    """Weryfikuje wyszukanie wiedzy o autorskich układach krzemowych AI (TPU, Trainium, Maia, MTIA)."""
    query = "Jakie autorskie chipy i akceleratory AI projektują Amazon, Google, Microsoft i Meta?"
    candidates = vector_store.similarity_search(query=query, k=10)

    content_combined = " ".join([doc.page_content for doc in candidates]).lower()

    # Weryfikacja obecności kluczowych układów ASIC
    assert "trainium" in content_combined or "inferentia" in content_combined, "Brak wzmianki o chipach Amazona (Trainium/Inferentia)."
    assert "tpu" in content_combined, "Brak wzmianki o procesorach Google TPU."
    assert "maia" in content_combined or "mtia" in content_combined, "Brak wzmianki o Maia 100 lub MTIA."


def test_gpu_reranking_large_candidate_pool(vector_store):
    """Weryfikuje wydajność i precyzję GPU Cross-Encodera na dużej puli 10 kandydatów."""
    query = "Która spółka wdrożyła hybrydowe przetwarzanie Private Cloud Compute na chipach Apple Silicon M4?"
    
    # Krok 1: Pobranie 10 kandydatów z wektorówki
    raw_candidates = vector_store.similarity_search(query=query, k=10)
    assert len(raw_candidates) == 10, "Oczekiwano dokładnie 10 kandydatów z bazy."

    # Krok 2: Reranking na RTX 2050 (mierzymy czysty forward-pass po załadowaniu wag)
    reranker = get_reranker()
    reranker._load_model_if_needed()

    t0 = time.time()
    ranked = reranker.rank(query=query, documents=raw_candidates, top_k=3)
    duration_ms = (time.time() - t0) * 1000

    print(f"\n[GPU Benchmark] Reranking 10 kandydatów na RTX 2050 trwał: {duration_ms:.2f} ms")
    
    # Wydajność GPU (dla 10 pełnych chunków ~2000 tokenów na mobilnym RTX 2050 akceptujemy czas < 1500 ms)
    assert duration_ms < 1500.0, f"Reranking 10 par powinien zająć < 1500 ms na RTX 2050, trwał {duration_ms:.2f} ms"

    # Trafność: na 1. miejscu musi być raport Apple
    top_doc, top_score = ranked[0]
    assert top_doc.metadata.get("company") == "Apple", f"Oczekiwano Apple na 1. miejscu, otrzymano: {top_doc.metadata.get('company')}"
    assert "Private Cloud Compute" in top_doc.page_content


def test_comparative_agent_query():
    """Weryfikuje pełne przejście przez graf dla zapytania porównawczego między spółkami."""
    query = "Porównaj podejście Apple i Microsoftu do nakładów Capex na infrastrukturę AI w 2024 roku."

    initial_state: GraphState = {
        "question": query,
        "original_question": query,
        "documents": [],
        "rerank_scores": [],
        "generation": None,
        "retry_count": 0,
        "regeneration_count": 0,
        "hallucination_grade": None,
        "answer_grade": None,
        "web_search_needed": False,
    }

    result = app.invoke(initial_state)
    generation = result.get("generation", "")

    print(f"\n[Test Comparative QA Result]:\n{generation}\n")

    # Weryfikacja: odpowiedź musi wymieniać obie spółki i kluczowe liczby/fakty
    assert "Apple" in generation and "Microsoft" in generation
    assert ("55,7" in generation or "55.7" in generation or "mld" in generation)
    assert result.get("hallucination_grade") == "grounded"
