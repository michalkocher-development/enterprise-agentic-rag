"""Zarządzanie bazą wektorową, trwałym cache'em dyskowym oraz Parent-Document Retrieval."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_settings
from src.ingestion.pdf_parser import FinancialPDFParser
from src.ingestion.hierarchical_chunker import HierarchicalChunker


class FinancialVectorStore:
    """Zarządca bazy wektorowej z obsługą Parent-Document Retrieval, PDF oraz Knowledge Lake."""

    def __init__(
        self,
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
        cache_file: str = "data/vector_cache/index.json",
    ) -> None:
        settings = get_settings()
        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
        self.cache_file = Path(cache_file)
        self.docstore_file = self.cache_file.parent / "parent_docstore.json"
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.parent_docstore: Dict[str, Document] = {}
        self.pdf_parser = FinancialPDFParser()
        self.hierarchical_chunker = HierarchicalChunker()
        self._is_populated: bool = False

    @staticmethod
    def _detect_company(file_path: Path) -> str:
        """Rozpoznaje firmę lub domenę na podstawie ścieżki i nazwy pliku."""
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
        elif "ai_act" in stem:
            return "EU Governance"
        return file_path.parent.name.capitalize() if file_path.parent.name != "financial_reports" else file_path.stem.capitalize()

    def save_to_disk(self) -> None:
        """Zapisuje indeks wektorowy dzieci oraz słownik nadrzędnych dokumentów (rodziców) na dysk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.dump(str(self.cache_file))

        # Zapis słownika rodziców (Parent Chunks)
        serialized_docstore = {
            pid: {"page_content": doc.page_content, "metadata": doc.metadata}
            for pid, doc in self.parent_docstore.items()
        }
        self.docstore_file.write_text(
            json.dumps(serialized_docstore, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_from_disk(self) -> bool:
        """Wczytuje zserializowany indeks wektorowy oraz magazyn rodziców z dysku."""
        if not self.cache_file.exists() or not self.docstore_file.exists():
            return False
        try:
            self.vector_store = InMemoryVectorStore.load(
                str(self.cache_file), embedding=self.embeddings
            )
            raw_docstore = json.loads(self.docstore_file.read_text(encoding="utf-8"))
            self.parent_docstore = {
                pid: Document(page_content=item["page_content"], metadata=item["metadata"])
                for pid, item in raw_docstore.items()
            }
            self._is_populated = True
            return True
        except Exception as e:
            print(f"Błąd ładowania cache'u z dysku: {e}. Indeks zostanie przebudowany.")
            return False

    def load_from_directory(
        self,
        reports_dir: str = "data/financial_reports",
        pdf_dir: str = "data/pdf_reports",
        kb_dir: str = "data/knowledge_base",
        force_reload: bool = False,
    ) -> int:
        """Wczytuje raporty finansowe, PDF oraz wielodomenowy Markdown Knowledge Lake."""
        if not force_reload and self.load_from_disk():
            # Sprawdzamy czy w knowledge_base są nowe pliki, których nie ma w docstore
            path_kb = Path(kb_dir)
            if path_kb.exists():
                indexed_sources = {d.metadata.get("source") for d in self.parent_docstore.values()}
                missing_files = [f for f in path_kb.rglob("*.md") if str(f) not in indexed_sources]
                if missing_files:
                    for mf in missing_files:
                        self.add_markdown_file(mf, domain=mf.parent.name)
            return -1

        raw_docs: List[Document] = []

        # 1. Odczyt raportów Markdown Big Tech
        path_md = Path(reports_dir)
        if path_md.exists():
            for file_path in sorted(path_md.glob("*.md")):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    company = self._detect_company(file_path)
                    raw_docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                                "company": company,
                                "is_table": False,
                            },
                        )
                    )
                except Exception as e:
                    print(f"Błąd odczytu pliku {file_path}: {e}")

        # 2. Odczyt i parsowanie sprawozdań PDF
        path_pdf = Path(pdf_dir)
        if path_pdf.exists():
            for file_path in sorted(path_pdf.glob("*.pdf")):
                try:
                    pdf_docs = self.pdf_parser.parse_pdf(file_path)
                    raw_docs.extend(pdf_docs)
                except Exception as e:
                    print(f"Błąd parsowania pliku PDF {file_path}: {e}")

        # 3. Odczyt wielodomenowego repozytorium wiedzy Knowledge Lake
        path_kb = Path(kb_dir)
        if path_kb.exists():
            for file_path in sorted(path_kb.rglob("*.md")):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    domain = file_path.parent.name
                    raw_docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                                "company": domain.capitalize(),
                                "domain": domain,
                                "title": file_path.stem.replace("_", " ").title(),
                                "is_table": "|" in content,
                            },
                        )
                    )
                except Exception as e:
                    print(f"Błąd odczytu z Knowledge Lake {file_path}: {e}")

        # 4. Hierarchiczny podział: Child Chunks i Parent Chunks
        children, parent_store = self.hierarchical_chunker.split_documents(raw_docs)
        self.vector_store.add_documents(children)
        self.parent_docstore = parent_store
        self._is_populated = True

        # 5. Zapisanie do trwałego cache'u
        if children:
            self.save_to_disk()

        return len(children)

    def add_markdown_file(self, file_path: Path, domain: str = "general") -> int:
        """Dynamicznie i przyrostowo indeksuje nowy plik z Knowledge Lake do bazy wektorowej."""
        file_path = Path(file_path)
        if not file_path.exists():
            return 0

        if not self._is_populated:
            self.load_from_directory()

        content = file_path.read_text(encoding="utf-8")
        raw_doc = Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "company": domain.capitalize(),
                "domain": domain,
                "title": file_path.stem.replace("_", " ").title(),
                "is_table": "|" in content,
            },
        )

        children, parent_store = self.hierarchical_chunker.split_documents([raw_doc])
        self.vector_store.add_documents(children)
        self.parent_docstore.update(parent_store)
        self.save_to_disk()
        print(f"Pomyślnie zindeksowano przyrostowo: {file_path.name} ({len(children)} chunków)")
        return len(children)

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """Parent-Document Retrieval: Wyszukuje wektorowo małe chunki dzieci,
        a zwraca unikalne, pełne dokumenty rodziców z bogatym kontekstem.
        """
        if not self._is_populated:
            self.load_from_directory()

        child_docs = self.vector_store.similarity_search(query, k=k * 2)

        unique_parents: List[Document] = []
        seen_parent_ids = set()

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_docstore:
                if parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    parent_doc = self.parent_docstore[parent_id]
                    unique_parents.append(parent_doc)
            else:
                if child.page_content not in [p.page_content for p in unique_parents]:
                    unique_parents.append(child)

            if len(unique_parents) >= k:
                break

        return unique_parents


_VECTOR_STORE_INSTANCE: Optional[FinancialVectorStore] = None


def get_vector_store() -> FinancialVectorStore:
    """Zwraca instancję singletona bazy wektorowej z automatycznym ładowaniem."""
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        _VECTOR_STORE_INSTANCE = FinancialVectorStore()
        _VECTOR_STORE_INSTANCE.load_from_directory()
    return _VECTOR_STORE_INSTANCE


def get_retriever() -> FinancialVectorStore:
    """Zwraca instancję bazy wektorowej do wyszukiwania (kompatybilność wsteczna)."""
    return get_vector_store()

