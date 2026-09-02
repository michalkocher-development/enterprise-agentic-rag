"""Zaawansowany parser plików PDF sprawozdań finansowych z analizą tabel (pdfplumber)."""

from pathlib import Path
from typing import List, Optional
import pdfplumber
from langchain_core.documents import Document


class FinancialPDFParser:
    """Ekstraktor tekstu i tabel z raportów finansowych w formacie PDF."""

    @staticmethod
    def _clean_cell(cell: Optional[str]) -> str:
        """Czyści zawartość komórki tabeli z nadmiarowych znaków nowej linii."""
        if not cell:
            return ""
        return " ".join(str(cell).split())

    @classmethod
    def table_to_markdown(cls, table: List[List[Optional[str]]]) -> str:
        """Konwertuje dwuwymiarową listę komórek wyciągniętą przez pdfplumber do formatu Markdown."""
        if not table or len(table) < 1:
            return ""

        cleaned_table: List[List[str]] = []
        for row in table:
            cleaned_row = [cls._clean_cell(cell) for cell in row]
            # Ignorowanie całkowicie pustych wierszy
            if any(cleaned_row):
                cleaned_table.append(cleaned_row)

        if not cleaned_table:
            return ""

        headers = cleaned_table[0]
        col_count = len(headers)

        lines: List[str] = []
        # Wiersz nagłówkowy
        lines.append("| " + " | ".join(headers) + " |")
        # Separator kolumn
        lines.append("| " + " | ".join(["---"] * col_count) + " |")

        # Wiersze z danymi
        for row in cleaned_table[1:]:
            # Wyrównanie liczby kolumn
            if len(row) < col_count:
                row.extend([""] * (col_count - len(row)))
            lines.append("| " + " | ".join(row[:col_count]) + " |")

        return "\n".join(lines)

    @staticmethod
    def _detect_company(file_path: Path) -> str:
        """Rozpoznaje spółkę na podstawie nazwy pliku."""
        stem = file_path.stem.lower()
        if "nvidia" in stem:
            return "NVIDIA"
        elif "alphabet" in stem or "google" in stem:
            return "Alphabet"
        elif "microsoft" in stem:
            return "Microsoft"
        elif "amazon" in stem or "aws" in stem:
            return "Amazon"
        elif "meta" in stem:
            return "Meta"
        elif "apple" in stem:
            return "Apple"
        return file_path.stem.capitalize()

    def parse_pdf(self, file_path: Path) -> List[Document]:
        """Parsuje plik PDF i zwraca listę dokumentów (osobno tekst i wyodrębnione tabele)."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Plik PDF nie istnieje: {file_path}")

        company = self._detect_company(file_path)
        documents: List[Document] = []

        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1

                # 1. Ekstrakcja tabel
                tables = page.extract_tables()
                for table_idx, raw_table in enumerate(tables):
                    md_table = self.table_to_markdown(raw_table)
                    if md_table:
                        table_content = f"### [Tabela Finansowa — Strona {page_num}, Tabela #{table_idx + 1}]\n{md_table}"
                        documents.append(
                            Document(
                                page_content=table_content,
                                metadata={
                                    "source": str(file_path),
                                    "filename": file_path.name,
                                    "company": company,
                                    "page": page_num,
                                    "content_type": "table",
                                    "is_table": True,
                                },
                            )
                        )

                # 2. Ekstrakcja tekstu ciągłego
                # Wyciągamy tekst z ignorowaniem stopek/nagłówków
                text = page.extract_text()
                if text:
                    # Filtrujemy powtarzające się klauzule formalne (np. stopki)
                    cleaned_lines = []
                    for line in text.splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.isdigit():  # sam numer strony
                            continue
                        cleaned_lines.append(stripped)

                    cleaned_text = "\n".join(cleaned_lines)
                    if cleaned_text:
                        documents.append(
                            Document(
                                page_content=cleaned_text,
                                metadata={
                                    "source": str(file_path),
                                    "filename": file_path.name,
                                    "company": company,
                                    "page": page_num,
                                    "content_type": "narrative_text",
                                    "is_table": False,
                                },
                            )
                        )

        return documents
