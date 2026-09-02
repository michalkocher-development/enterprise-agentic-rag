"""Uniwersalny normalizator dokumentów: konwersja PDF, skanów (OCR) i TXT do strukturyzowanego Markdownu."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pypdfium2
import pdfplumber
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from src.ingestion.pdf_parser import FinancialPDFParser


class NormalizedDocument:
    """Reprezentacja znormalizowanego dokumentu w formacie Markdown z metadanymi."""

    def __init__(
        self,
        title: str,
        domain: str,
        content: str,
        metadata: Dict[str, Any],
        tables: Optional[List[str]] = None,
    ) -> None:
        self.title = title
        self.domain = domain
        self.content = content
        self.metadata = metadata
        self.tables = tables or []

    def to_markdown_with_frontmatter(self) -> str:
        """Generuje pełny dokument Markdown z nagłówkiem YAML Frontmatter."""
        yaml_lines = [
            "---",
            f"title: \"{self.title}\"",
            f"domain: \"{self.domain}\"",
            f"doc_type: \"{self.metadata.get('doc_type', 'document')}\"",
            f"source_file: \"{self.metadata.get('source_file', '')}\"",
            f"pages_count: {self.metadata.get('pages_count', 1)}",
            f"tables_count: {len(self.tables)}",
            f"ocr_used: {str(self.metadata.get('ocr_used', False)).lower()}",
            f"processed_at: \"{self.metadata.get('processed_at', datetime.now().isoformat())}\"",
            "---",
            "",
            f"# {self.title}",
            "",
            self.content,
        ]
        return "\n".join(yaml_lines)


class DocumentNormalizer:
    """Hybrydowy konwerter plików (cyfrowy PDF, skany OCR, obrazy, TXT) do czystego Markdownu."""

    def __init__(self) -> None:
        self._ocr: Optional[RapidOCR] = None
        self.table_formatter = FinancialPDFParser.table_to_markdown

    @property
    def ocr(self) -> RapidOCR:
        """Lazy initialization silnika OCR (oszczędność pamięci)."""
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

    def _ocr_image(self, img: Image.Image) -> str:
        """Przeprowadza OCR na obiekcie PIL Image i zwraca odczytany tekst."""
        results, _ = self.ocr(img)
        if not results:
            return ""
        # results to lista: [ [box, text, score], ... ]
        extracted_lines = [item[1] for item in results if len(item) > 1 and item[1].strip()]
        return "\n".join(extracted_lines)

    def normalize_txt(self, file_path: Path, domain: str = "general") -> NormalizedDocument:
        """Normalizuje plik tekstowy (.txt)."""
        content = file_path.read_text(encoding="utf-8", errors="replace").strip()
        title = file_path.stem.replace("_", " ").title()

        return NormalizedDocument(
            title=title,
            domain=domain,
            content=content,
            metadata={
                "doc_type": "text",
                "source_file": file_path.name,
                "pages_count": 1,
                "ocr_used": False,
                "processed_at": datetime.now().isoformat(),
            },
        )

    def normalize_image(self, file_path: Path, domain: str = "general") -> NormalizedDocument:
        """Normalizuje plik graficzny (.png, .jpg, .jpeg) za pomocą silnika OCR."""
        with Image.open(file_path) as img:
            ocr_text = self._ocr_image(img)

        title = file_path.stem.replace("_", " ").title()
        return NormalizedDocument(
            title=title,
            domain=domain,
            content=f"*(Treść wyodrębniona za pomocą OCR)*\n\n{ocr_text}",
            metadata={
                "doc_type": "scanned_image",
                "source_file": file_path.name,
                "pages_count": 1,
                "ocr_used": True,
                "processed_at": datetime.now().isoformat(),
            },
        )

    def normalize_pdf(self, file_path: Path, domain: str = "general") -> NormalizedDocument:
        """Normalizuje plik PDF z hybrydową detekcją:
        - dla stron cyfrowych: bezstratna ekstrakcja tekstu i tabel do Markdown,
        - dla stron skanowanych (brak tekstu): automatyczny fallback na OCR (pypdfium2 + RapidOCR).
        """
        file_path = Path(file_path)
        all_tables: List[str] = []
        page_sections: List[str] = []
        ocr_was_used = False

        # 1. Analiza strukturalna i ekstrakcja tabel przez pdfplumber
        with pdfplumber.open(file_path) as pdf:
            pdfium_doc = pypdfium2.PdfDocument(str(file_path))
            total_pages = len(pdf.pages)

            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                page_content_parts: List[str] = [f"## Strona {page_num}"]

                # Ekstrakcja tabel
                tables = page.extract_tables()
                page_has_tables = False
                for t_idx, raw_table in enumerate(tables):
                    md_table = self.table_formatter(raw_table)
                    if md_table:
                        page_has_tables = True
                        all_tables.append(md_table)
                        page_content_parts.append(
                            f"### Tabela Finansowa #{len(all_tables)}\n{md_table}"
                        )

                # Ekstrakcja tekstu cyfrowego
                text = page.extract_text() or ""
                cleaned_text = "\n".join(
                    [line.strip() for line in text.splitlines() if line.strip() and not line.strip().isdigit()]
                )

                # Sprawdzenie czy strona jest skanem (mało tekstu i brak tabel)
                if len(cleaned_text) < 40 and not page_has_tables:
                    # Fallback na OCR renderując stronę do obrazu przez pypdfium2
                    pdfium_page = pdfium_doc[page_idx]
                    pil_image = pdfium_page.render(scale=2.0).to_pil()
                    ocr_result = self._ocr_image(pil_image)
                    if ocr_result:
                        ocr_was_used = True
                        page_content_parts.append(f"*(Skan strony — odczytano przez OCR)*:\n{ocr_result}")
                else:
                    if cleaned_text:
                        page_content_parts.append(cleaned_text)

                page_sections.append("\n\n".join(page_content_parts))

        combined_content = "\n\n---\n\n".join(page_sections)
        title = file_path.stem.replace("_", " ").title()

        return NormalizedDocument(
            title=title,
            domain=domain,
            content=combined_content,
            metadata={
                "doc_type": "pdf",
                "source_file": file_path.name,
                "pages_count": total_pages,
                "ocr_used": ocr_was_used,
                "processed_at": datetime.now().isoformat(),
            },
            tables=all_tables,
        )

    def normalize_file(self, file_path: Path, domain: str = "general") -> NormalizedDocument:
        """Główny punkt wejścia routera normalizacyjnego."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".txt":
            return self.normalize_txt(file_path, domain=domain)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            return self.normalize_image(file_path, domain=domain)
        elif ext == ".pdf":
            return self.normalize_pdf(file_path, domain=domain)
        else:
            raise ValueError(f"Nieobsługiwany format pliku: {ext}")

    def save_to_knowledge_lake(
        self,
        doc: NormalizedDocument,
        base_dir: str = "data/knowledge_base",
    ) -> Path:
        """Zapisuje znormalizowany dokument jako czytelny plik .md w odpowiednim podkatalogu domeny."""
        target_dir = Path(base_dir) / doc.domain
        target_dir.mkdir(parents=True, exist_ok=True)

        slug = "".join([c if c.isalnum() or c in "-_" else "_" for c in doc.title.lower()]).strip("_")
        target_file = target_dir / f"{slug}.md"

        full_markdown = doc.to_markdown_with_frontmatter()
        target_file.write_text(full_markdown, encoding="utf-8")
        return target_file
