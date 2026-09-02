"""Testy jednostkowe i integracyjne dla Modułu 2 (Retriever) oraz Modułu 1 + 2 (Retriever + Reranker)."""

import pytest
from src.config import get_settings
from src.reranker.local_reranker import get_reranker
from src.retriever.vector_store import FinancialVectorStore, get_vector_store


@pytest.fixture
def vector_store():
    """Zwraca instancję bazy wektorowej załadowanej raportami finansowymi."""
    return get_vector_store()


def test_embedding_dimensions(vector_store):
    """Weryfikuje połączenie z Gemini Embeddings API i wymiarowość embeddingów."""
    query = "Przychody segmentu sztucznej inteligencji"
    embedding = vector_store.embeddings.embed_query(query)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 3072, f"Oczekiwano wymiaru 3072 dla modelu Gemini, otrzymano {len(embedding)}"


def test_document_ingestion_and_retrieval(vector_store):
    """Weryfikuje poprawne wczytanie raportów i wyszukanie kandydatów dla NVIDIA Data Center."""
    query = "Jakie były przychody segmentu Data Center NVIDIA w Q3 FY2025?"
    candidates = vector_store.similarity_search(query=query, k=4)

    assert len(candidates) > 0, "Baza wektorowa powinna zwrócić co najmniej 1 fragment."
    
    # Przynajmniej jeden z fragmentów powinien dotyczyć NVIDIA i zawierać wzmiankę o Data Center
    found_nvidia = any("nvidia" in doc.page_content.lower() for doc in candidates)
    found_datacenter = any("data center" in doc.page_content.lower() for doc in candidates)
    
    assert found_nvidia, "Wśród zwróconych kandydatów powinien znaleźć się raport NVIDIA."
    assert found_datacenter, "Wśród zwróconych kandydatów powinna znaleźć się informacja o Data Center."


def test_pipeline_retriever_plus_reranker(vector_store):
    """Weryfikuje pełny potok: Gęste wyszukiwanie wektorowe + Lokalny PyTorch Reranker na RTX 2050."""
    query = "Ile wynosiły łączne nakłady inwestycyjne Capex spółki Alphabet w całym 2024 roku?"

    # Krok 1: Wyszukanie wstępne w bazie wektorowej (Gemini Embeddings)
    raw_candidates = vector_store.similarity_search(query=query, k=6)
    assert len(raw_candidates) >= 2, "Retriever powinien zwrócić co najmniej 2 kandydatów."

    # Krok 2: Precyzyjny reranking na lokalnym GPU RTX 2050 (Moduł 1)
    reranker = get_reranker()
    ranked_results = reranker.rank(query=query, documents=raw_candidates, top_k=2)

    assert len(ranked_results) == 2
    top_doc, top_score = ranked_results[0]

    print(f"\n[Test Pipeline] Najwyżej oceniony fragment (Score: {top_score:.4f}):\n{top_doc.page_content[:200]}...")

    # Weryfikacja: najwyżej oceniony dokument musi zawierać precyzyjną kwotę 51,4 mld USD Capex Alphabetu
    assert "51,4 mld USD" in top_doc.page_content or "51,4" in top_doc.page_content, (
        f"Lokalny Cross-Encoder powinien wytypować fragment z kwotą 51,4 mld USD! Treść: {top_doc.page_content}"
    )
    assert top_doc.metadata.get("company") == "Alphabet"
