"""Testy jednostkowe lokalnego rerankera PyTorch na RTX 2050."""

import pytest
import torch
from langchain_core.documents import Document

from src.reranker.local_reranker import LocalCrossEncoderReranker, get_reranker


@pytest.fixture
def reranker():
    """Zwraca instancję singletona rerankera."""
    return get_reranker()


def test_reranker_singleton_and_device(reranker):
    """Weryfikuje wzorzec singletona i poprawne przypisanie GPU."""
    another_instance = get_reranker()
    assert reranker is another_instance, "Reranker musi być pojedynczą instancją (Singleton)."
    
    if torch.cuda.is_available():
        assert reranker.device == "cuda", "Na maszynie z CUDA reranker powinien działać na 'cuda'."


def test_reranker_ranking_accuracy(reranker):
    """Weryfikuje, czy model Cross-Encoder poprawnie pozycjonuje relewantny fragment na 1. miejscu."""
    query = "Jaki model i limit pamięci VRAM posiada lokalna karta graficzna?"

    doc_target = Document(
        page_content="Lokalna stacja robocza posiada dedykowaną kartę graficzną NVIDIA GeForce RTX 2050 wyposażoną w 4096 MiB pamięci VRAM.",
        metadata={"id": "target"},
    )
    doc_distractor_1 = Document(
        page_content="Google Gemini 3.0 Flash to model językowy w chmurze służący do rozumowania i syntezy tekstu.",
        metadata={"id": "distractor_1"},
    )
    doc_distractor_2 = Document(
        page_content="Usługa LocalStack pozwala na bezkosztową emulację storage'u AWS S3 na porcie 4566.",
        metadata={"id": "distractor_2"},
    )

    documents = [doc_distractor_1, doc_target, doc_distractor_2]

    results = reranker.rank(query=query, documents=documents, top_k=3)

    assert len(results) == 3
    top_doc, top_score = results[0]

    # Sprawdzenie, czy najbardziej relewantny dokument wygrał
    assert top_doc.metadata["id"] == "target", f"Oczekiwano dokumentu target na szczycie, otrzymano: {top_doc.metadata['id']}"
    
    # Sprawdzenie monotoniczności ocen
    assert results[0][1] >= results[1][1] >= results[2][1], "Wyniki powinny być posortowane malejąco wg ocen."
    
    # Oceny powinny być w przedziale (0, 1) dzięki funkcji sigmoid
    for doc, score in results:
        assert 0.0 <= score <= 1.0, f"Score {score} poza zakresem [0, 1]."


def test_reranker_vram_efficiency(reranker):
    """Weryfikuje, czy model mieści się w rygorystycznym budżecie VRAM (< 600 MB)."""
    if torch.cuda.is_available():
        vram_mb = reranker.get_vram_usage_mb()
        print(f"\n[VRAM Monitor] Zużycie pamięci VRAM przez Cross-Encoder: {vram_mb:.2f} MB")
        
        # RTX 2050 ma 4096 MB, nasz model bge-reranker-base w FP16 powinien zająć < 600 MB
        assert vram_mb < 600.0, f"Zużycie VRAM ({vram_mb:.2f} MB) przekracza dopuszczalny limit 600 MB!"


def test_reranker_edge_cases(reranker):
    """Weryfikuje zachowanie przy pustej liście dokumentów."""
    assert reranker.rank(query="Dowolne pytanie", documents=[], top_k=3) == []
