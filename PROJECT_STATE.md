# AI Engineering & LLMOps Lab — Rejestr Stanu Projektu (Project State)

> **Rola pliku**: To jest główne repozytorium pamięci operacyjnej projektu. Agent i deweloper zaglądają tu na początku i na końcu każdej sesji, aby zachować 100% spójności kontekstu bez polegania na zawodnej pamięci krótkotrwałej czatu.

---

## 1. Bieżący Status
* **Status Ogólny**: 🟢 **Projekt 1 ukończony z sukcesem!**
* **Aktywny Projekt**: **Projekt 1 — Self-Corrective Agentic RAG z PyTorch Rerankerem i LangSmith**.
* **Aktualny Krok**: Pełny potok agentowy przetestowany jednostkowo i integracyjnie (10/10 testów na zielono), CLI `src/run_agent.py` w pełni funkcjonalne z tracingiem w LangSmith. Gotowy do ewaluacji jakościowej (Moduł 5) lub przejścia do Projektu 2 (MLOps & Local AWS).

---

## 2. Mapa Drogowa (Roadmap)

### Etap 0: Fundamenty i Zarządzanie Kontekstem (Zakończony ✅)
- [x] Utworzenie dedykowanego katalogu roboczego (`ai-engineering-lab`).
- [x] Opracowanie instrukcji bazowych w `GEMINI.md`.
- [x] Utworzenie rejestru stanu i decyzji w `PROJECT_STATE.md`.
- [x] Utworzenie szablonu `.env.example` i `.gitignore`.
- [x] Konfiguracja pliku `.env` z kluczami API (Gemini & LangSmith).
- [x] Utworzenie wirtualnego środowiska `.venv` (Python 3.12).
- [x] Weryfikacja akceleracji PyTorch CUDA na RTX 2050 i połączenia z Gemini API.

### Etap 1: Projekt 1 — Self-Corrective Agentic RAG (Zakończony ✅)
- [x] Moduł 1: PyTorch Inference Node — ładowanie Cross-Encodera (`bge-reranker-base`) na RTX 2050 w FP16 (Singleton, testy jednostkowe na zielono).
- [x] Moduł 2: Retriever & Vector Store (InMemoryVectorStore z Gemini Embeddings `gemini-embedding-001`).
- [x] Moduł 3 & 4: LangGraph State Machine & Agentic Nodes (`retrieve`, `local_rerank`, `grade_documents`, `rewrite_query`, `generate`, `hallucination_check` + automatyczny tracing do LangSmith).
- [x] Moduł 5: Testy regresyjne i integracyjne (14/14 testów na zielono, interaktywne CLI `src/run_agent.py`).
- [x] Skalowanie Bazy Wiedzy: Pełna „Wielka Szóstka” Big Tech (NVIDIA, Alphabet, Microsoft, Amazon AWS, Meta, Apple) + Trwały dyskowy cache wektorowy (`data/vector_cache/index.json`, ładowanie w 0.5s).

### Etap 2: Projekt 2 — MLOps, Hybrydowy Routing i Local AWS (LocalStack)
- [ ] Konfiguracja LocalStack / mock S3 pod zapis wektorów i checkpointów modeli.
- [ ] Lekki fine-tuning PyTorch (klasyfikator intencji zapytania na HuggingFace).
- [ ] Zbudowanie pipeline'u automatycznego pobierania wag z S3 (boto3).
- [ ] Spięcie routera w LangGraph.

### Etap 3: Projekt 3 — Capstone: Eval-Driven Enterprise Engine
- [ ] Zestaw testów regresyjnych (CI/CD eval suite) w LangSmith.
- [ ] Optymalizacja PyTorch (quantization / torch.compile).
- [ ] Konteneryzacja i raport końcowy.

---

## 3. Profil Środowiska i Sprzętu
| Element | Konfiguracja | Uwagi |
| :--- | :--- | :--- |
| **System Operacyjny** | Windows 11 (PowerShell) | Ścieżki w Windows format lub forward slash |
| **GPU** | NVIDIA GeForce RTX 2050 (4 GB VRAM) | CUDA 13.1, Driver 592.00 |
| **Python** | 3.12.0 | Wirtualne środowisko: `.venv` |
| **LLM Provider** | Google Gemini | Klucz API Gemini (wymagany w `.env`) |
| **Observability** | LangSmith | Klucz API LangSmith (wymagany w `.env`) |
| **Cloud Storage** | LocalStack / moto / boto3 | Bezkosztowa symulacja AWS S3 |

---

## 4. Architectural Decision Records (ADR)

### ADR-001: Strategia zarządzania pamięcią agenta
* **Data**: 2026-09-02
* **Kontekst**: W dużych projektach AI / LLMOps okno kontekstowe LLM szybko ulega degradacji, a sesje w Antigravity bywają restartowane.
* **Decyzja**: Wykorzystanie dwuwarstwowego systemu pamięci:
  1. `GEMINI.md` w workspace root jako trwała reguła wstrzykiwana do każdego zapytania.
  2. `PROJECT_STATE.md` jako żywy dokument stanu (Living Document), aktualizowany synchronicznie przed i po każdym zadaniu.
* **Konsekwencje**: Zapewnia zerową utratę kontekstu między sesjami oraz natychmiastowe przywracanie stanu prac.

### ADR-002: Wybór środowiska chmurowego (Brak konta AWS)
* **Data**: 2026-09-02
* **Kontekst**: Użytkownik nie posiada konta ani budżetu na AWS, a celem jest nauka standardów chmurowych w ML/LLMOps.
* **Decyzja**: Zastosowanie **LocalStack** (lub biblioteki `moto` w testach jednostkowych) wraz z oficjalnym SDK `boto3`.
* **Konsekwencje**: Kod aplikacji (`boto3.client('s3', ...)`) jest w 100% zgodny z produkcyjnym AWS. Jedyną zmienną konfiguracyjną jest endpoint URL w `.env`. Brak jakichkolwiek opłat czy ryzyka wycieku kluczy chmurowych.

### ADR-003: Alokacja VRAM dla PyTorch na RTX 2050
* **Data**: 2026-09-02
* **Kontekst**: Karta RTX 2050 posiada 4096 MiB VRAM. Duże modele językowe generujące (np. 7B/8B) zajęłyby cały VRAM lub wywołały Out-Of-Memory (OOM).
* **Decyzja**: Podział odpowiedzialności (Hybrid Compute Architecture):
  - Ciężkie generowanie tekstu i rozumowanie: Google Gemini API (w chmurze).
  - Precyzyjny scoring, reranking fragmentów i lokalna inferencja: PyTorch na RTX 2050 z modelami Cross-Encoder w precyzji `torch.float16` (zużycie VRAM: < 500 MB).
* **Konsekwencje**: Maksymalna wydajność, brak ryzyka OOM, zerowy koszt tokenów za reranking setek fragmentów tekstu.

### ADR-004: Foundation Model — Gemini 3.0 Flash
* **Data**: 2026-09-02
* **Kontekst**: Potrzebujemy szybkiego, precyzyjnego modelu do oceny dokumentów (grading), wykrywania halucynacji (groundedness) oraz syntezy końcowej odpowiedzi.
* **Decyzja**: Wybór modelu `gemini-3.0-flash` jako domyślnego LLM w całym potoku LangGraph.
* **Konsekwencje**: Bardzo niska latencja generowania, wysokie okno kontekstowe i precyzyjne strukturyzowane wyjścia (Pydantic / Function Calling).

### ADR-005: Domena Bazy Wiedzy — Sprawozdawczość Finansowa i Nakłady na AI (Big Tech)
* **Data**: 2026-09-02
* **Kontekst**: Potrzebujemy wymagającej domeny tekstowej z dużą gęstością liczb, okresów rozliczeniowych i subtelnych rozróżnień semantycznych do przetestowania odporności na halucynacje.
* **Decyzja**: Wybór raportów finansowych spółek technologicznych (np. NVIDIA, Alphabet — raporty roczne 10-K / kwartalne 10-Q, ze szczególnym uwzględnieniem przychodów Data Center, nakładów Capex na AI oraz czynników ryzyka).
* **Konsekwencje**: Idealny poligon dla Self-Corrective RAG: zwykły RAG gubi się w kwartałach i wskaźnikach, podczas gdy Cross-Encoder precyzyjnie identyfikuje właściwy okres i liczbę, a Hallucination Checker weryfikuje ich zgodność.

### ADR-006: Asynchroniczna współbieżność ewaluacji (AsyncIO Parallelization)
* **Data**: 2026-09-02
* **Kontekst**: Węzły `grade_documents` i `hallucination_check` wykonywały zapytania do Gemini API w pętli sekwencyjnej (4 chunki = ~10-12s, plus sprawdzanie halucynacji = ~8-9s), co powodowało długie czasy oczekiwania na odpowiedź użytkownika.
* **Decyzja**: Wdrożenie asynchronicznej współbieżności przez `asyncio.gather` i `ainvoke` z dedykowanym wrapperem `run_async` w `src/utils/async_runner.py`:
  1. Wszystkie kandydackie chunki w `grade_documents` ewaluowane są jednocześnie w jednym locie HTTP/2.
  2. Niezależne testy Groundedness oraz Answer Relevance w `hallucination_node` wykonują się w pełni równolegle.
* **Konsekwencje**: Czas wykonania testów regresyjnych spadł ze 106s do 63s (~41% redukcji całkowitego czasu), a pojedyncze zapytanie analityczne działa ponad 2x szybciej przy zerowym wzroście kosztu tokenów.


