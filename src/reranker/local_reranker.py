"""Lokalny reranker Cross-Encoder z akceleracją PyTorch CUDA na NVIDIA GeForce RTX 2050."""

import math
from typing import List, Optional, Tuple
import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import get_settings


class LocalCrossEncoderReranker:
    """Zoptymalizowany pod kątem VRAM lokalny Cross-Encoder (Singleton).

    Obsługuje modele takie jak BAAI/bge-reranker-base na lokalnym GPU z precyzją
    torch.float16, gwarantując minimalny narzut pamięciowy i natychmiastową
    inferencję.
    """

    _instance: Optional["LocalCrossEncoderReranker"] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_name
        self.device = device or settings.get_resolved_device()
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.reranker_fp16

        self._model: Optional[CrossEncoder] = None

    @classmethod
    def get_instance(cls) -> "LocalCrossEncoderReranker":
        """Zwraca lub tworzy singleton instancji rerankera."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model_if_needed(self) -> CrossEncoder:
        """Lazy loading wag modelu do pamięci GPU."""
        if self._model is None:
            # Określenie typu danych (FP16 na CUDA dla oszczędności VRAM)
            torch_dtype = torch.float16 if (self.use_fp16 and self.device == "cuda") else torch.float32

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                model_kwargs={"torch_dtype": torch_dtype} if self.device == "cuda" else {},
            )
        return self._model

    @staticmethod
    def _sigmoid(logit: float) -> float:
        """Przelicza surowy logit modelu na prawdopodobieństwo z zakresu [0.0, 1.0]."""
        return 1.0 / (1.0 + math.exp(-logit))

    def rank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
    ) -> List[Tuple[Document, float]]:
        """Dokonuje precyzyjnego rerankingu listy dokumentów względem zadanego pytania.

        Args:
            query: Pytanie użytkownika.
            documents: Lista dokumentów (kandydatów z bazy wektorowej).
            top_k: Liczba najlepszych dokumentów do zwrócenia.

        Returns:
            Lista krotek (Dokument, score) posortowana malejąco wg trafności.
        """
        if not documents:
            return []

        model = self._load_model_if_needed()

        # Budowanie par (query, doc_text) dla Cross-Encodera
        pairs = [(query, doc.page_content) for doc in documents]

        with torch.inference_mode():
            # Predykcja logitów podobieństwa
            raw_scores = model.predict(pairs)

            # Konwersja do float w pythonie
            if hasattr(raw_scores, "tolist"):
                scores_list = raw_scores.tolist()
            elif isinstance(raw_scores, (list, tuple)):
                scores_list = list(raw_scores)
            else:
                scores_list = [float(raw_scores)]

            # Normalizacja logitów przez sigmoida
            normalized_scores = [self._sigmoid(float(s)) for s in scores_list]

        # Zwolnienie pamięci podręcznej alokatora PyTorch
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Skojarzenie dokumentów z ocenami
        doc_scores = list(zip(documents, normalized_scores))

        # Sortowanie malejąco wg score'a
        doc_scores.sort(key=lambda item: item[1], reverse=True)

        return doc_scores[:top_k]

    def get_vram_usage_mb(self) -> float:
        """Zwraca aktualnie zaalokowaną pamięć VRAM w megabajtach (MB)."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0


def get_reranker() -> LocalCrossEncoderReranker:
    """Fabryka / getter dla instancji Singletona rerankera."""
    return LocalCrossEncoderReranker.get_instance()
