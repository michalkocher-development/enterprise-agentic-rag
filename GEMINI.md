# AI Engineering & LLMOps Lab — Antigravity Workspace Rules

## 1. Cel Projektu & Profil
Praktyczne laboratorium inżynierii AI / LLMOps:
- **Orkiestracja**: LangGraph, cykliczne grafy stanów, Agentic RAG (CRAG, Self-RAG).
- **ML / Deep Learning**: PyTorch z akceleracją CUDA (RTX 2050 4GB VRAM) – własne rerankery, embeddingi, lekki fine-tuning / adaptery.
- **LLM / Foundation Models**: Google Gemini API (`langchain-google-genai` / Google GenAI SDK).
- **LLMOps & Observability**: LangSmith (tracing, metryki latencji, evals).
- **Cloud / MLOps**: LocalStack / moto / boto3 (symulacja AWS S3 / IAM / storage bez kosztów chmurowych).

---

## 2. Pamięć i Zarządzanie Kontekstem (Context Management Protocol)
Każda interakcja w tym projekcie musi szanować poniższe zasady zachowania ciągłości:

1. **Źródło Prawdy (Single Source of Truth)**:
   - Plik `PROJECT_STATE.md` jest naszą żywą pamięcią. Zawsze przed rozpoczęciem i po zakończeniu zadania weryfikujemy i aktualizujemy ten plik.
2. **Architektura Decyzji (ADR - Architecture Decision Records)**:
   - Każda istotna decyzja technologiczna (wybór modelu, dobór struktury grafu, format danych) musi być odnotowana w `PROJECT_STATE.md` w sekcji ADR wraz z uzasadnieniem.
3. **Zasada Progressive Disclosure**:
   - Zamiast wrzucać do promptu całe kody źródłowe, utrzymujemy zwięzłe interfejsy i dokumentację modułów.
   - Do badań i czytania długich dokumentacji używamy subagentów (`research`), aby nie zaśmiecać głównego okna kontekstowego.
4. **Planning Mode First**:
   - Przy zadaniach architektonicznych i nowych modułach najpierw tworzymy plan (`implementation_plan.md`), czekamy na akceptację, a po wykonaniu podsumowujemy w `walkthrough.md`.

---

## 3. Ograniczenia Sprzętowe i Środowisko
- **OS**: Windows 11 / PowerShell.
- **GPU**: NVIDIA GeForce RTX 2050 (4096 MiB VRAM).
  - *Reguła VRAM*: Unikać modeli > 1B parametrów lokalnie. Do rerankingu stosować modele typu Cross-Encoder `bge-reranker-base` (~110M) lub `ms-marco-MiniLM` z precyzją `FP16` lub `BF16`.
  - Zawsze zwalniać pamięć (`torch.cuda.empty_cache()`) po cięższych operacjach testowych.
- **Python**: 3.12 (używamy wirtualnego środowiska `.venv`).

---

## 4. Konwencje Kodu i Jakości
- Pełne typowanie (type hinting) we wszystkich węzłach LangGraph (`TypedDict` / Pydantic `BaseModel`).
- Każdy węzeł grafu musi być przetestowany jednostkowo zanim zostanie wpięty do głównego `StateGraph`.
- Klucze API wyłącznie przez plik `.env` (nigdy w kodzie źródłowym ani repozytorium).
