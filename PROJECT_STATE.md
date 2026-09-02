# AI Engineering & LLMOps Lab — Rejestr Stanu Projektu (Project State)

> **Rola pliku**: To jest główne repozytorium pamięci operacyjnej projektu. Agent i deweloper zaglądają tu na początku i na końcu każdej sesji, aby zachować 100% spójności kontekstu bez polegania na zawodnej pamięci krótkotrwałej czatu.

---

## 1. Bieżący Status
* **Status Ogólny**: 🟢 **Nowe repozytorium utworzone: `enterprise-agentic-rag`!**
* **Aktywny Projekt**: **Enterprise Agentic Document Intelligence & RAG Platform**.
* **Repozytorium GitHub**: [https://github.com/michalkocher-development/enterprise-agentic-rag](https://github.com/michalkocher-development/enterprise-agentic-rag)
* **Aktualny Krok**: Utworzono niezależny silnik normalizacji dokumentów `DocumentNormalizer` z obsługą tabel (`pdfplumber`), skanów OCR (`RapidOCR`) i zapisem do Markdown Knowledge Lake. Testy (3/3) na zielono. Ready for Fast API & Memory!

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

### ADR-007: Ekstrakcja tabel finansowych z PDF i Parent-Document Retrieval
* **Data**: 2026-09-02
* **Kontekst**: Tradycyjny RAG zawodzi na surowych sprawozdaniach finansowych w PDF: tabele są rozcinane na bezwartościowe fragmenty, a stały chunking (np. 700 znaków) tworzy dylemat między precyzją wyszukania a pełnią kontekstu dla modelu.
* **Decyzja**:
  1. Zastosowanie biblioteki `pdfplumber` z analizą siatki komórek do bezstratnej konwersji tabel bilansowych do formatu Markdown (`| Kategoria | Wartość |`).
  2. Implementacja wzorca **Parent-Document Retrieval** (`HierarchicalChunker`): drobne *Child Chunks* (~250 zn.) indeksowane są w wektorówce jako precyzyjne haczyki wyszukiwawcze, po czym retriever automatycznie podmienia je na pełne *Parent Chunks* (~1200 zn. / całe tabele) z pamięci `parent_docstore`.
* **Konsekwencje**: Eliminacja rozmycia semantycznego, chirurgiczna precyzja odnajdywania liczb w tabelach i zachowanie 100% kontekstu dla sędziego halucynacji i syntezy LLM. Indeks powiększony do 18 testów (100% passed).

### ADR-008: Interfejs Chatbota w Gradio i Deployment na Hugging Face Spaces
* **Data**: 2026-09-02
* **Kontekst**: Potrzebny jest przyjazny, estetyczny interfejs webowy demonstrujący działanie grafu LangGraph, podgląd cytowanych tabel oraz darmowy hosting w chmurze bez ponoszenia kosztów AWS.
* **Decyzja**:
  1. Wybór biblioteki **Gradio 6** (`app.py` w `gr.Blocks`) ze streamingiem kroków węzłów agenta i dwukolumnowym układem (Czat + Inspektor LLMOps).
  2. Przygotowanie projektu pod **Hugging Face Spaces** (darmowy tier 16 GB RAM / 2 vCPU):
     - Automatyczny fallback inferencji PyTorch: CUDA na maszynie deweloperskiej, CPU w chmurze HF Spaces.
     - Metadane YAML w `README.md`.
* **Konsekwencje**: Pełna wizualizacja procesu myślenia agenta na żywo, natychmiastowe uruchomienie aplikacji z dyskowego cache'u i możliwość 1-kliknięciowego udostępnienia działającego demo w chmurze. Zestaw testów rozszerzony do 21/21 passed.

### ADR-009: Dwufazowy potok Ingestion (Normalizer z OCR ➔ Markdown Knowledge Lake ➔ Baza Wektorowa)
* **Data**: 2026-09-02
* **Kontekst**: Bezpośrednie indeksowanie chaotycznych plików PDF i skanów do bazy wektorowej powodowało rozmycie semantyczne i brak możliwości audytu tego, co agent "wie".
* **Decyzja**:
  1. Wdrożenie klasy `DocumentNormalizer` z hybrydową ekstrakcją:
     - Dla PDF cyfrowych: `pdfplumber` (tabele do Markdown + czysty tekst),
     - Dla skanów i obrazów: silnik OCR `RapidOCR` (`rapidocr-onnxruntime`) działający w pamięci bez zewnętrznych instalatorów binarnych C++.
  2. Zapisywanie znormalizowanych dokumentów jako pliki `.md` z nagłówkiem YAML Frontmatter w strukturze `data/knowledge_base/<domena>/<plik>.md`.
  3. Baza wektorowa indeksuje czysty Markdown, co daje wyższą jakość embeddingów i pełną transparentność dla dewelopera/użytkownika.
* **Konsekwencje**: 100% audytowalność bazy wiedzy w Git, obsługa skanów/obrazów/PDF/TXT, moduł w pełni przetestowany niezależnie (`test_document_normalizer.py`).

### ADR-010: Pamięć Konwersacyjna, Adaptive RAG i Rozproszony Serwer FastAPI
* **Data**: 2026-09-02
* **Kontekst**: Potrzebna była obsługa dialogu wieloturowego (pamięć wcześniejszych pytań bez marnowania tokenów i czasu na niepotrzebny RAG), profesjonalne REST API ze Swagger UI (`/docs`) oraz niezależny frontend webowy bez narzutów Gradio.
* **Decyzja**:
  1. Wdrożenie `MemorySaver` w LangGraph wraz z węzłem `route_question_node`: dopytania użytkownika nawiązujące do historii dialogu są obsługiwane bezpośrednio (`direct_answer`), a nowy RAG odpala się tylko przy pytaniach o nowe fakty.
  2. Budowa serwera FastAPI z modelami Pydantic v2, endpointem uploadu i OCR (`/api/v1/documents/upload`), streamingiem zdarzeń węzłów przez Server-Sent Events (`/api/v1/chat/stream`) oraz interaktywną dokumentacją Swagger pod `/docs`.
  3. Serwowanie dedykowanego frontendu HTML5 + Tailwind CSS a la Perplexity bezpośrednio na ścieżce głównej `/`.
* **Konsekwencje**: Pełna separacja warstw (REST API vs Frontend), pamięć konwersacyjna (`thread_id`), streaming SSE, 30/30 testów regresyjnych na zielono.




