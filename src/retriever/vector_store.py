"""Zarządzanie bazą wektorową, trwałym cache'em dyskowym i pobieraniem kandydatów (Dense Retrieval)."""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


class FinancialVectorStore:
    """Zarządca bazy wektorowej dla sprawozdań finansowych Big Tech z trwałym cache'owaniem."""

    def __init__(
        self,
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        cache_file: str = "data/vector_cache/index.json",
    ) -> None:
        settings = get_settings()
        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
        self.cache_file = Path(cache_file)
        self.vector_store = InMemoryVectorStore(self.embeddings)
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

    def ingest_documents(self, documents: List[Document]) -> int:
        """Dzieli dokumenty na chunki i dodaje do indeksu wektorowego."""
        if not documents:
            return 0
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)
        self._is_populated = True
        return len(chunks)

    def save_to_disk(self) -> None:
        """Zapisuje bieżący indeks wektorowy na dysk w formacie JSON."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.dump(str(self.cache_file))

    def load_from_disk(self) -> bool:
        """Wczytuje zserializowany indeks wektorowy z dysku."""
        if not self.cache_file.exists():
            return False
        try:
            self.vector_store = InMemoryVectorStore.load(
                str(self.cache_file), embedding=self.embeddings
            )
            self._is_populated = True
            return True
        except Exception as e:
            print(f"Błąd ładowania cache'u z dysku: {e}. Indeks zostanie przebudowany.")
            return False

    def load_from_directory(
        self,
        directory_path: str = "data/financial_reports",
        force_reload: bool = False,
    ) -> int:
        """Wczytuje sprawozdania finansowe. Używa cache'u dyskowego, jeśli istnieje."""
        # 1. Próba załadowania z cache'u dyskowego
        if not force_reload and self.load_from_disk():
            return -1  # Oznaczenie, że wczytano z cache

        path = Path(directory_path)
        if not path.exists():
            return 0

        raw_docs: List[Document] = []
        for file_path in sorted(path.glob("*.md")):
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
                        },
                    )
                )
            except Exception as e:
                print(f"Błąd odczytu pliku {file_path}: {e}")

        chunk_count = self.ingest_documents(raw_docs)
        if chunk_count > 0:
            self.save_to_disk()

        return chunk_count

    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Wyszukuje top-k najbardziej zbliżonych semantycznie fragmentów."""
        if not self._is_populated:
            self.load_from_directory()
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self, k: int = 8):
        """Zwraca obiekt retrievera z domyślnym rozmiarem puli kandydatów."""
        if not self._is_populated:
            self.load_from_directory()
        return self.vector_store.as_retriever(search_kwargs={"k": k})


_global_vector_store: Optional[FinancialVectorStore] = None


def get_vector_store(force_reload: bool = False) -> FinancialVectorStore:
    """Zwraca globalną instancję bazy wektorowej (Singleton / cache)."""
    global _global_vector_store
    if _global_vector_store is None or force_reload:
        _global_vector_store = FinancialVectorStore()
        _global_vector_store.load_from_directory(force_reload=force_reload)
    return _global_vector_store


def get_retriever(k: int = 8):
    """Zwraca skonfigurowany obiekt retrievera."""
    return get_vector_store().get_retriever(k=k)
