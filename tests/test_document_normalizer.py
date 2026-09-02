"""Niezależne testy jednostkowe i integracyjne dla DocumentNormalizer z obsługą OCR i Markdown Lake."""

from pathlib import Path
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ingestion.document_normalizer import DocumentNormalizer, NormalizedDocument


@pytest.fixture
def normalizer():
    return DocumentNormalizer()


@pytest.fixture
def temp_lake(tmp_path):
    lake_dir = tmp_path / "knowledge_base"
    lake_dir.mkdir(parents=True, exist_ok=True)
    return lake_dir


def test_normalize_txt_file(normalizer, tmp_path, temp_lake):
    """Weryfikuje konwersję surowego pliku .txt do usystematyzowanego Markdownu z metadanymi YAML."""
    raw_txt = tmp_path / "eu_ai_act_summary.txt"
    raw_txt.write_text(
        "Unijny Akt o Sztucznej Inteligencji (EU AI Act) klasyfikuje systemy na 4 poziomy ryzyka: "
        "nieakceptowalne, wysokie, ograniczone i minimalne. Kary za naruszenia wynoszą do 35 mln EUR lub 7% rocznego obrotu.",
        encoding="utf-8",
    )

    doc = normalizer.normalize_file(raw_txt, domain="governance")
    assert doc.title == "Eu Ai Act Summary"
    assert doc.domain == "governance"
    assert "35 mln EUR" in doc.content

    saved_path = normalizer.save_to_knowledge_lake(doc, base_dir=str(temp_lake))
    assert saved_path.exists()
    assert saved_path.suffix == ".md"

    md_content = saved_path.read_text(encoding="utf-8")
    assert "---" in md_content
    assert 'domain: "governance"' in md_content
    assert "# Eu Ai Act Summary" in md_content


def test_normalize_pdf_with_tables(normalizer, temp_lake):
    """Weryfikuje konwersję wielostronicowego PDF ze sprawozdaniem do ustrukturyzowanego pliku Markdown."""
    pdf_path = Path("data/pdf_reports/nvidia_q3_fy25_10q.pdf")
    assert pdf_path.exists(), "Testowy plik PDF musi istnieć"

    doc = normalizer.normalize_file(pdf_path, domain="finance")
    assert doc.domain == "finance"
    assert len(doc.tables) >= 2
    assert "| Total Revenue |" in doc.content
    assert "$ 35,082" in doc.content

    saved_path = normalizer.save_to_knowledge_lake(doc, base_dir=str(temp_lake))
    assert saved_path.exists()

    md_content = saved_path.read_text(encoding="utf-8")
    assert "tables_count: 2" in md_content or "tables_count: " in md_content
    assert "## Strona 1" in md_content


def test_normalize_image_with_ocr(normalizer, tmp_path, temp_lake):
    """Weryfikuje działanie silnika OCR na wygenerowanym syntetycznym skanie dokumentu."""
    # Tworzymy syntetyczny obraz testowy ze stemplem i tekstem
    img_path = tmp_path / "scanned_invoice.png"
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Rysujemy wyraźny tekst symulujący zeskanowany nagłówek
    draw.text((20, 40), "FAKTURA VAT: FV/2026/09/01", fill=(0, 0, 0))
    draw.text((20, 80), "KWOTA NETTO: 45000 PLN", fill=(0, 0, 0))
    draw.text((20, 120), "STATUS: OPLACONE", fill=(0, 0, 0))
    img.save(img_path)

    doc = normalizer.normalize_file(img_path, domain="invoices")
    assert doc.metadata["ocr_used"] is True
    assert "FAKTURA" in doc.content or "45000" in doc.content or "OPLACONE" in doc.content

    saved_path = normalizer.save_to_knowledge_lake(doc, base_dir=str(temp_lake))
    assert saved_path.exists()

    md_content = saved_path.read_text(encoding="utf-8")
    assert 'ocr_used: true' in md_content
    assert 'domain: "invoices"' in md_content
