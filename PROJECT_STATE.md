# AI Engineering & LLMOps Lab — Rejestr Stanu Projektu (Project State)

> **Rola pliku**: To jest główne repozytorium pamięci operacyjnej projektu. Agent i deweloper zaglądają tu na początku i na końcu każdej sesji, aby zachować 100% spójności kontekstu bez polegania na zawodnej pamięci krótkotrwałej czatu.

---

## 1. Bieżący Status
* **Status Ogólny**: 🟢 **Nowoczesny Workbench UI (100% Live Mode) zaimplementowany i wdrożony!**
* **Aktywny Projekt**: **Enterprise Agentic Document Intelligence & RAG Platform**.
* **Repozytorium GitHub**: [https://github.com/michalkocher-development/enterprise-agentic-rag](https://github.com/michalkocher-development/enterprise-agentic-rag)
* **Aktualny Krok**: Wdrożono nowy frontend Workbench UI (`static/index.html`) inspirowany projektem UX Designera: 100% tryb na żywo (SSE), animowany graf SVG stanów, streaming tokenów w Markdown, 4 pełne karty z pytaniami testowymi w głównym widoku oraz zintegrowany moduł wgrywania dokumentów i OCR.


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

### ADR-011: Multi-Stage Telemetry, Werdykty Gradera i Dynamiczna Ingestia Knowledge Lake
* **Data**: 2026-09-02
* **Kontekst**: Wcześniejsza wektorówka wymagała restartu serwera do wczytania nowych plików z Knowledge Lake, a UI nie pokazywało szczegółowych decyzji sędziego merytorycznego ani pętli samonaprawczych (rewrite query), gdy żaden chunk nie pasował.
* **Decyzja**:
  1. Wdrożenie `add_markdown_file()` w `FinancialVectorStore`: przyrostowy chunking, wektoryzacja i zasilenie pamięci `parent_docstore` oraz zapisu cache w czasie rzeczywistym.
  2. Rozbudowa stanu grafu i węzła `grade_documents`: model zwraca ustrukturyzowany werdykt (`relevant` / `irrelevant`) wraz ze zwięzłym uzasadnieniem słownym dla każdego ocenianego chunku.
  3. Uogólnienie węzła `rewrite_node` do uniwersalnego optymalizatora zapytań we wszystkich domenach (dyplomy, prawo, finanse, badania).
  4. Dodanie automatycznej synchronizacji zmiennych środowiskowych `os.environ` dla natywnego śledzenia LangSmith.
* **Konsekwencje**: 100% natychmiastowa dostępność wgrywanych dokumentów, pełna transparentność decyzji modelu dla użytkownika, odporność na pętle samonaprawcze.

### ADR-012: Interaktywny Eksplorator Bazy Wiedzy (Knowledge Lake Explorer) w UI i API
* **Data**: 2026-09-02
* **Kontekst**: Użytkownik potrzebuje bezpośredniego podglądu z poziomu UI, jakie dokumenty znajdują się w bazie wiedzy, ile zawierają tabel, czy przeszły przez OCR oraz możliwości przeczytania ich pełnej treści przed zadaniem pytania.
* **Decyzja**:
  1. Wdrożenie endpointu `GET /api/v1/documents/{domain}/{filename}` zwracającego pełną treść wybranego pliku Markdown.
  2. Dodanie w UI przycisku nagłówka `Baza Dokumentów` z licznikiem na żywo oraz dedykowanej zakładki `📁 Baza Wiedzy` w prawym panelu analitycznym.
  3. Stworzenie interaktywnego modala z filtrami domenowymi (`Finanse`, `Edukacja / OCR`, `Prawo`), wyszukiwarką czasu rzeczywistego oraz akcjami `🔍 Podgląd treści` i `💬 Zadaj pytanie o ten dokument`.
* **Konsekwencje**: Kompletny, zintegrowany interfejs Knowledge Lake, gdzie użytkownik może badać i weryfikować zbiór wiedzy bez opuszczania aplikacji.

### ADR-013: Protokół Telemetryczny SSE, Replay Engine i Gotowość pod Nowy Workbench UI
* **Data**: 2026-09-02
* **Kontekst**: Projektant UI przedstawił specyfikację nowego Workbench UI (`from_UX_designer/NOTES.md`), wymagającego:
  1. Natychmiastowego startu węzła (`node_start`) z kierunkiem krawędzi (`edge_from`) do animacji krawędzi SVG,
  2. Streamingu tokenów (`token`) w węźle generowania,
  3. Stabilnych identyfikatorów fragmentów (`chunk_id`) łączących cytowania, wiersze rerankingu i werdykty gradera,
  4. Scoringu wszystkich 10 kandydatów z flagą `kept` do animacji przetasowania,
  5. Metody HTTP GET dla standardowego `EventSource`,
  6. Deterministycznego silnika odtwarzania scenariuszy (`/api/v1/replay/{run_id}`) na potrzeby bezkosztowej prezentacji portfolio rekruterom,
  7. Rozgrzewki GPU w `lifespan` FastAPI i heartbeat keepalive (`: keepalive`).
* **Decyzja**:
  1. Wdrożenie `astream_events(version="v2")` w `StatefulAgentGraph` i centralnym generatorze `generate_chat_sse`.
  2. Emisja zdarzeń: `session`, `node_start`, `token`, `step`, `error`, `complete` oraz cyklicznego keepalive.
  3. Generowanie stabilnych `chunk_id` (`{stem}#p{idx}` / `{stem}#t{idx}`) w `HierarchicalChunker` i propagowanie ich przez wszystkie węzły.
  4. Pełne sortowanie wszystkich kandydatów w `rerank_node` z polami `score` i `kept: bool`.
  5. Dodanie modułu `src/api/replays.py` z 3 scenariuszami (`direct_answer`, `standard_rag`, `self_correction`) i endpointami `/api/v1/replays` oraz `/api/v1/replay/{run_id}`.
  6. Wdrożenie `lifespan` z warmupem Cross-Encodera oraz natywna obsługa `GET /api/v1/chat/stream`.
* **Konsekwencje**: Pełna gotowość backendu do natychmiastowego spięcia z nowym frontendem, 32/32 testów regresyjnych na zielono (100% passed), zerowe ryzyko cold-startu i zerowy koszt prezentacji demo dzięki trybowi Replay.

### ADR-014: Nowoczesny Workbench UI (100% Live Mode, Ekran Pytań Testowych i Ingestia Dokumentów)
* **Data**: 2026-09-02
* **Kontekst**: Wdrożenie interfejsu zaprojektowanego przez UX Designera (`agentic-rag-workbench.html`), z uwzględnieniem specyfiki produkcyjnej: całkowite wyeliminowanie mocków/demo (100% czysty tryb live oparty na Server-Sent Events z backendem), dodanie brakującego w projekcie modułu wgrywania dokumentów i OCR oraz wyeksponowanie pełnych treści pytań analitycznych z ich uzasadnieniem i metadanymi.
* **Decyzja**:
  1. Pełna implementacja estetyki Dark Minimalist Workbench (`static/index.html`) z paletą barw designera (`#08090A`, `#0D0F10`, `#14171A`), responsywnym układem `100vh` na desktopie i animowanym grafem SVG (8 węzłów, przepływy krawędzi, pętla autokorekty).
  2. Architektura 100% Live: bez mocków, bezpośrednia integracja z `EventSource` na `/api/v1/chat/stream?q=...` odbierająca zdarzenia `session`, `node_start`, `token` (ze streamingiem markdown przez Marked.js), `step`, `error`, `complete`.
  3. Prezentacja pytań testowych i ergonomia stopki: w głównym stanie początkowym wyeksponowano 4 estetyczne karty zawierające samą treść pytań, a w stopce wprowadzono zoptymalizowany, jednoliniowy układ 2-kolumnowy (szybki wybór pod oknem odpowiedzi, a pasek wpisywania pod kolumną telemetryczną), co pozwoliło na powiększenie grafu SVG stanów do 215px.
  4. Moduł Ingestii i Zarządzania Dokumentami: zintegrowano modal uploadu z OCR i Parent-Document chunkingiem (`/api/v1/documents/upload`) oraz eksplorator bazy wiedzy (`/api/v1/documents`) z możliwością natychmiastowego zadania pytania o wybrany plik.
  5. Audyt interfejsu i Observability: wyeliminowano kolizje etykiet grafu SVG – wydłużono krawędź e6 do 68px dla etykiety 'trafne', poszerzono pętlę e10 (regeneracja dołem z 135px do 205px: M1070 V168 H865), zapewniając 28px marginesu wokół grota strzałki 'halucynacja (regeneracja)', a w nagłówku i endpointzie `/api/v1/health` przywrócono widoczność i weryfikację aktywnego tracera LangSmith (`langsmith_active: true`).
  6. Protokół Bezpieczeństwa Treści (Gatekeeper przed Halucynacją): odpowiedź generowana przez LLM jest buforowana w pamięci i nie jest ujawniana użytkownikowi, dopóki węzeł `hallucination_check` nie wyda pozytywnego werdyktu ugruntowania w zweryfikowanych źródłach (`grounded` / `yes`). W razie halucynacji brudnopis jest bezpowrotnie odrzucany i uruchamiana jest regeneracja, chroniąc użytkownika przed dezinformacją.
  7. Bezpośrednie linki do narzędzi (Swagger UI i LangSmith) i pełna interaktywność dokumentów źródłowych: na górnym pasku w miejsce wskaźnika GPU umieszczono bezpośredni odnośnik do interaktywnej dokumentacji OpenAPI/Swagger (`/docs`), zachowano link do projektu w chmurze LangSmith (`smith.langchain.com`), a w interfejsie dodano wielopoziomowe otwieranie treści dokumentów (klikalny tytuł pliku `citeDetailTitleBtn`, przycisk `Cały plik`, chipy cytowań, odnośniki inline `[1]`, `[2]` w treści odpowiedzi oraz modal pełnego podglądu pliku Markdown `docPreviewModal` ze 100% precyzyjnym mapowaniem domen).
* **Konsekwencje**: Pełna ochrona wiarygodności faktograficznej platformy, natychmiastowy dostęp do telemetrii w LangSmith i dokumentacji Swagger UI oraz możliwość dogłębnej inspekcji źródeł i tabel źródłowych przez analityka bez opuszczania interfejsu.

### ADR-015: Scalenie Gałęzi Wizualnej (ui/workbench-visual-redesign) do Głównej Linii (main)
* **Data**: 2026-09-02
* **Kontekst**: Połączenie zaawansowanych usprawnień wizualnych interfejsu (taśma przebiegu pod grafem z segmentami proporcjonalnymi do czasu wykonania i rozróżnieniem chmury vs lokalnego GPU, typografia IBM Plex / Archivo, kaskada cytowań, animacja wycieraczkowa ujawniania odpowiedzi oraz obsługa skrótów klawiaturowych) z kompletnym silnikiem backendowym i telemetrią na gałęzi `main`.
* **Decyzja**: Scalenie gałęzi `ui/workbench-visual-redesign` (commit `1c005cd`) do `main` w trybie fast-forward.
* **Konsekwencje**: Główna linia projektu `main` posiada zunifikowany, dopracowany wizualnie i w 100% sprawny technicznie interfejs badawczy AI Engineering / LLMOps Lab.

### ADR-016: Produkcyjna Konteneryzacja Docker i Wdrożenie na Hugging Face Spaces
* **Data**: 2026-09-02
* **Kontekst**: Wdrożenie platformy w chmurze bez ponoszenia opłat serwerowych (eliminacja kosztów AWS GPU/CPU) przy jednoczesnym zachowaniu wysokiej wydajności (16 GB RAM) i natychmiastowego czasu odpowiedzi.
* **Decyzja**:
  1. Wybór darmowej infrastruktury **Hugging Face Spaces (Docker SDK)** oferującej bezpłatnie **2 vCPU i 16 GB RAM**.
  2. Utworzenie zoptymalizowanego `Dockerfile` na bazie `python:3.12-slim-bookworm` z instalacją PyTorch w wydaniu CPU (`--index-url https://download.pytorch.org/whl/cpu`), co zredukowało rozmiar obrazu o ~3.5 GB.
  3. Pre-cache wag modelu Cross-Encoder (`BAAI/bge-reranker-base`) podczas budowania obrazu, eliminując opóźnienia cold-start przy uruchamianiu kontenera.
  4. Skonfigurowanie dedykowanego użytkownika `user` (UID 1000) oraz portu `7860` zgodnie ze standardami bezpieczeństwa Hugging Face Spaces.
  5. Zaktualizowanie metadanych `README.md` (YAML frontmatter dla Spaces) oraz `.dockerignore`.
* **Konsekwencje**: Kompletna, przenośna konteneryzacja umożliwiająca uruchomienie platformy Enterprise Agentic RAG jednym poleceniem na Hugging Face Spaces lub dowolnym środowisku Docker za 0 zł.




