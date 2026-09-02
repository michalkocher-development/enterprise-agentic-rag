"""Modele danych Pydantic v2 dla serwera REST API i dokumentacji OpenAPI / Swagger."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Żądanie zapytania analitycznego do agenta."""

    question: str = Field(
        ...,
        min_length=2,
        description="Treść zapytania użytkownika (np. pytanie o zarobki, artykuł EU AI Act, lub dopytanie w dialogu).",
        examples=["Ile wynosił zysk netto NVIDIA w Q3 FY2025 według oficjalnego raportu 10-Q?"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Identyfikator wątku konwersacji (umożliwia zachowanie pamięci wieloturowej). Jeśli brak, zostanie wygenerowany.",
        examples=["session-4819"],
    )
    domain: Optional[str] = Field(
        default=None,
        description="Opcjonalne zawężenie domeny przeszukiwania (np. 'finance', 'governance', 'education', 'ai_research').",
        examples=["finance"],
    )
    lang: str = Field(
        default="pl",
        description="Język odpowiedzi oraz ewaluacji ('pl' dla polskiego, 'en' dla angielskiego).",
        examples=["pl"],
    )


class StepLog(BaseModel):
    """Zapis pojedynczego kroku wykonania cyklicznego grafu LangGraph."""

    node: str = Field(..., description="Nazwa węzła grafu (np. router, retrieve, local_rerank, generate).")
    description: str = Field(..., description="Opis czynności wykonanej przez węzeł.")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Metryki lub parametry wygenerowane przez węzeł.")


class Citation(BaseModel):
    """Cytowane źródło ugruntowujące odpowiedź agenta."""

    chunk_id: Optional[str] = Field(default=None, description="Stabilny identyfikator fragmentu przez cały przebieg.")
    source: str = Field(..., description="Ścieżka lub identyfikator źródła.")
    filename: str = Field(..., description="Nazwa pliku źródłowego.")
    company_or_domain: str = Field(..., description="Wykryta spółka lub domena merytoryczna.")
    content_snippet: str = Field(..., description="Fragment tekstu lub tabeli Markdown stanowiący podstawę faktu.")
    is_table: bool = Field(default=False, description="Czy fragment jest tabelą finansową/strukturalną.")


class ChatResponse(BaseModel):
    """Pełna odpowiedź analityczna z audytem wykonania i cytatami."""

    answer: str = Field(..., description="Wygenerowana, ugruntowana w faktach odpowiedź analityczna.")
    thread_id: str = Field(..., description="Identyfikator wątku dla zachowania ciągłości pamięci.")
    route_taken: str = Field(..., description="Kierunek wyznaczony przez Router ('retrieve' lub 'direct_answer').")
    steps_log: List[StepLog] = Field(default_factory=list, description="Ślad decyzyjny grafu stanów LangGraph.")
    citations: List[Citation] = Field(default_factory=list, description="Lista cytowanych fragmentów i tabel.")
    hallucination_grade: Optional[str] = Field(default=None, description="Wynik audytu ugruntowania faktograficznego.")
    latency_ms: float = Field(..., description="Całkowity czas przetwarzania zapytania w milisekundach.")


class IngestResponse(BaseModel):
    """Wynik przetworzenia i normalizacji przesłanego dokumentu."""

    success: bool = Field(..., description="Status operacji.")
    filename: str = Field(..., description="Nazwa przetworzonego pliku.")
    domain: str = Field(..., description="Przypisana domena w Knowledge Lake.")
    title: str = Field(..., description="Wykryty lub nadany tytuł dokumentu.")
    markdown_path: str = Field(..., description="Ścieżka do wygenerowanego pliku Markdown z metadanymi.")
    pages_count: int = Field(..., description="Liczba przetworzonych stron.")
    tables_count: int = Field(..., description="Liczba bezbłędnie wyekstrahowanych tabel Markdown.")
    ocr_used: bool = Field(..., description="Czy do odczytania treści zastosowano silnik OCR.")
    message: str = Field(..., description="Komunikat statusu.")


class DocumentItem(BaseModel):
    """Informacja o dokumencie znajdującym się w Markdown Knowledge Lake."""

    filename: str = Field(..., description="Nazwa pliku Markdown.")
    domain: str = Field(..., description="Domena (podkatalog).")
    title: str = Field(..., description="Tytuł z nagłówka YAML.")
    relative_path: str = Field(..., description="Ścieżka względna.")
    size_bytes: int = Field(..., description="Rozmiar pliku na dysku.")
    tables_count: int = Field(default=0, description="Liczba tabel.")
    ocr_used: bool = Field(default=False, description="Czy plik powstał ze skanu OCR.")


class HealthResponse(BaseModel):
    """Raport stanu zdrowia i diagnostyki sprzętowej serwisu."""

    status: str = Field(default="healthy", description="Status działania serwera.")
    device: str = Field(..., description="Aktywne urządzenie inferencyjne ('cuda' lub 'cpu').")
    gpu_ready: bool = Field(default=True, description="Czy akcelerator GPU jest zainicjalizowany i gotowy.")
    model_warmed_up: bool = Field(default=True, description="Czy wagi modelu rerankera są rozgrzane w pamięci.")
    langsmith_active: bool = Field(default=False, description="Czy aktywny jest natywny tracing LangSmith.")
    gpu_name: Optional[str] = Field(default=None, description="Model karty GPU (jeśli dostępny).")
    vram_allocated_mb: Optional[float] = Field(default=None, description="Aktualnie zaalokowana pamięć VRAM.")
    vector_cache_loaded: bool = Field(..., description="Czy baza wektorowa i docstore są załadowane do pamięci.")
    knowledge_base_documents: int = Field(..., description="Liczba plików w Markdown Knowledge Lake.")


class ReplayScenarioItem(BaseModel):
    """Metadane zapisanego scenariusza demo do odtwarzania w trybie portfolio."""

    id: str = Field(..., description="Identyfikator scenariusza (np. direct_answer, standard_rag, self_correction).")
    title: str = Field(..., description="Czytelny tytuł scenariusza.")
    description: str = Field(..., description="Opis przebiegu i demonstrowanych możliwości.")
    query: str = Field(..., description="Zapytanie użytkownika.")
    has_self_correction: bool = Field(default=False, description="Czy scenariusz demonstruje pętlę samonaprawczą.")
    estimated_duration_s: float = Field(..., description="Szacowany czas trwania scenariusza w sekundach.")
