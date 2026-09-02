"""Testy integracyjne dla modułu parsowania PDF z pdfplumber oraz Parent-Document Retrieval."""

from pathlib import Path
import pytest
from src.graph import app
from src.ingestion.pdf_parser import FinancialPDFParser
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.retriever.vector_store import get_vector_store
from src.state import GraphState


@pytest.fixture
def pdf_path():
    path = Path("data/pdf_reports/nvidia_q3_fy25_10q.pdf")
    assert path.exists(), f"Plik testowy PDF nie istnieje: {path}"
    return path


def test_pdf_table_extraction_integrity(pdf_path):
    """Weryfikuje, czy tabele finansowe z PDF są bezbłędnie wyciągane jako tabele Markdown."""
    parser = FinancialPDFParser()
    docs = parser.parse_pdf(pdf_path)

    table_docs = [d for d in docs if d.metadata.get("is_table")]
    assert len(table_docs) >= 2, f"Oczekiwano co najmniej 2 tabel finansowych w 10-Q, znaleziono: {len(table_docs)}"

    # Weryfikacja pierwszej tabeli (Statements of Operations)
    t1_content = table_docs[0].page_content
    assert "| Total Revenue |" in t1_content
    assert "$ 35,082" in t1_content
    assert "$ 19,309" in t1_content  # Net Income

    # Weryfikacja drugiej tabeli (Market Platforms)
    t2_content = table_docs[1].page_content
    assert "| Data Center |" in t2_content
    assert "$ 30,771 M" in t2_content
    assert "$ 449 M" in t2_content  # Automotive


def test_hierarchical_chunking_parent_child_link(pdf_path):
    """Weryfikuje relację Child -> Parent w hierarchicznym podziale dokumentów."""
    parser = FinancialPDFParser()
    raw_docs = parser.parse_pdf(pdf_path)

    chunker = HierarchicalChunker()
    children, docstore = chunker.split_documents(raw_docs)

    assert len(children) > len(docstore), "Liczba Child Chunks powinna być większa niż liczba Parent Chunks."

    for child in children:
        parent_id = child.metadata.get("parent_id")
        assert parent_id is not None, "Child Chunk musi posiadać przypisane parent_id."
        assert parent_id in docstore, f"Parent ID {parent_id} musi istnieć w docstore."

        parent = docstore[parent_id]
        # Sprawdzenie czy dziecko jest podzbiorem lub logiczną częścią rodzica
        assert len(parent.page_content) >= len(child.page_content)


def test_parent_retrieval_from_table_query():
    """Weryfikuje czy zapytanie o komórkę tabeli zwraca cały spójny nadrzędny Parent Chunk."""
    vs = get_vector_store()
    query = "Ile wynosiły przychody platformy Automotive & Robotics w Q3 FY2025 według raportu NVIDIA?"

    retrieved_parents = vs.similarity_search(query=query, k=2)
    assert len(retrieved_parents) >= 1

    combined_content = "\n".join([p.page_content for p in retrieved_parents])
    # Sprawdzamy, czy w zwróconych pełnych rodzicach znajduje się kwota 449 oraz kontekst platform
    assert "449" in combined_content
    assert "Automotive" in combined_content
    assert ("Data Center" in combined_content or "Gaming" in combined_content)


def test_end_to_end_agent_on_pdf_table():
    """Weryfikuje pełne przejście przez graf agenta dla pytania opartego o tabelę z raportu PDF."""
    query = "Ile wyniósł zysk netto (Net Income) spółki NVIDIA w Q3 FY2025 według oficjalnego raportu 10-Q?"

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

    print(f"\n[Test PDF Agent Answer]:\n{generation}\n")

    # Weryfikacja: zysk netto 19,309 mld USD / 19 309 mln USD musi zostać poprawnie zacytowany z tabeli
    assert "19 309" in generation or "19,309" in generation or "19.309" in generation or "19,3" in generation
    assert result.get("hallucination_grade") == "grounded"
