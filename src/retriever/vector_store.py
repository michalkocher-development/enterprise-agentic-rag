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
    """Zarządca bazy wektorowej dla raportów finansowych z obsługą Parent-Document Retrieval i PDF."""

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
        """Rozpoznaje firmę na podstawie nazwy pliku."""
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
        force_reload: bool = False,
    ) -> int:
        """Wczytuje raporty Markdown oraz PDF, wykonując hierarchiczny chunking i zapisując cache."""
        if not force_reload and self.load_from_disk():
            return -1  # Oznaczenie wczytania z gotowego cache'u

        raw_docs: List[Document] = []

        # 1. Odczyt raportów Markdown
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

        # 2. Odczyt i parsowanie sprawozdań PDF (z ekstrakcją tabel do Markdown)
        path_pdf = Path(pdf_dir)
        if path_pdf.exists():
            for file_path in sorted(path_pdf.glob("*.pdf")):
                try:
                    pdf_docs = self.pdf_parser.parse_pdf(file_path)
                    raw_docs.extend(pdf_docs)
                except Exception as e:
                    print(f"Błąd parsowania pliku PDF {file_path}: {e}")

        # 3. Hierarchiczny podział: Child Chunks (wyszukiwanie) i Parent Chunks (kontekst)
        children, parent_store = self.hierarchical_chunker.split_documents(raw_docs)
        self.vector_store.add_documents(children)
        self.parent_docstore = parent_store
        self._is_populated = True

        # 4. Zapisanie do trwałego cache'u na dysku
        if children:
            self.save_to_disk()

        return len(children)

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """Parent-Document Retrieval: Wyszukuje wektorowo małe chunki dzieci,
        a zwraca unikalne, pełne dokumenty rodziców z bogatym kontekstem.
        """
        if not self._is_populated:
            self.load_from_directory()

        # Pobieramy nieco więcej dzieci, aby po deduplikacji rodziców mieć dokładnie k wyników
        child_docs = self.vector_store.similarity_search(query, k=k * 2)

        unique_parents: List[Document] = []
        seen_parent_ids = set()

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_docstore:
                if parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    unique_parents.append(self.parent_docstore[parent_id])
            else:
                # Dokument nieposiadający rodzica (fallback)
                unique_parents.append(child)

            if len(unique_parents) >= k:
                break

        return unique_parents

    def get_retriever(self, k: int = 10):
        """Zwraca interfejs kompatybilny z retrieverem LangChain."""
        if not self._is_populated:
            self.load_from_directory()
        return self.vector_store.as_retriever(search_kwargs={"k": k})


_global_vector_store: Optional[FinancialVectorStore] = None


def get_vector_store(force_reload: bool = False) -> FinancialVectorStore:
    """Zwraca globalną instancję bazy wektorowej."""
    global _global_vector_store
    if _global_vector_store is None or force_reload:
        _global_vector_store = FinancialVectorStore()
        _global_vector_store.load_from_directory(force_reload=force_reload)
    return _global_vector_store


def get_retriever(k: int = 10):
    """Zwraca skonfigurowany obiekt retrievera."""
    return get_vector_store().get_retriever(k=k)
