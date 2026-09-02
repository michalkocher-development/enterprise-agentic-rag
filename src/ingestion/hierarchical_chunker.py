"""Hierarchiczny chunker (Parent-Document Chunker) łączący precyzyjne przeszukiwanie z pełnym kontekstem."""

import uuid
from typing import Dict, List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HierarchicalChunker:
    """Implementacja podziału hierarchicznego: Child Chunks (wyszukiwanie) -> Parent Chunks (kontekst)."""

    def __init__(
        self,
        parent_chunk_size: int = 1200,
        parent_chunk_overlap: int = 150,
        child_chunk_size: int = 250,
        child_chunk_overlap: int = 40,
    ) -> None:
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n", ". ", "? ", "! ", " "],
        )

    def split_documents(
        self, documents: List[Document]
    ) -> Tuple[List[Document], Dict[str, Document]]:
        """Dzieli listę dokumentów na małe fragmenty wyszukiwawcze (dzieci) i słownik pełnych kontekstów (rodzice).
        
        Tabele finansowe są traktowane jako niepodzielne jednostki atomowe.
        """
        child_documents: List[Document] = []
        parent_docstore: Dict[str, Document] = {}

        for doc in documents:
            filename = doc.metadata.get("filename", "doc")
            file_stem = filename.rsplit(".", 1)[0]
            is_table = doc.metadata.get("is_table", False)

            if is_table:
                # 1. Tabele finansowe: Cała tabela jest jednostką nadrzędną (Parent)
                p_idx = len(parent_docstore)
                chunk_id = f"{file_stem}#t{p_idx}"
                parent_id = str(uuid.uuid4())
                parent_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "parent_id": parent_id, "chunk_id": chunk_id, "is_parent": True},
                )
                parent_docstore[parent_id] = parent_doc

                # Dzieci dla tabeli: dzielimy wierszami lub małym splitterem pod zapytania komórkowe
                child_splits = self.child_splitter.split_text(doc.page_content)
                for c_idx, split_text in enumerate(child_splits):
                    child_doc = Document(
                        page_content=split_text,
                        metadata={
                            **doc.metadata,
                            "parent_id": parent_id,
                            "chunk_id": chunk_id,
                            "child_id": f"{chunk_id}c{c_idx}",
                            "is_child": True,
                        },
                    )
                    child_documents.append(child_doc)

            else:
                # 2. Tekst ciągły (narrative text): Podział na Parent Chunks
                parent_splits = self.parent_splitter.split_text(doc.page_content)

                for p_idx, p_text in enumerate(parent_splits):
                    chunk_id = f"{file_stem}#p{p_idx}"
                    parent_id = str(uuid.uuid4())
                    parent_doc = Document(
                        page_content=p_text,
                        metadata={**doc.metadata, "parent_id": parent_id, "chunk_id": chunk_id, "is_parent": True},
                    )
                    parent_docstore[parent_id] = parent_doc

                    # Podział rodzica na drobne Child Chunks
                    child_splits = self.child_splitter.split_text(p_text)
                    for c_idx, c_text in enumerate(child_splits):
                        child_doc = Document(
                            page_content=c_text,
                            metadata={
                                **doc.metadata,
                                "parent_id": parent_id,
                                "chunk_id": chunk_id,
                                "child_id": f"{chunk_id}c{c_idx}",
                                "is_child": True,
                            },
                        )
                        child_documents.append(child_doc)

        return child_documents, parent_docstore

