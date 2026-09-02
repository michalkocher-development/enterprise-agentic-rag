# Schemat Przepływu Danych — Enterprise Agentic RAG

Niniejszy dokument przedstawia pełny schemat architektury i przepływu danych w platformie **Enterprise Agentic RAG**. Obejmuje on potok ingestii dokumentów, hierarchiczne wyszukiwanie Parent-Document, cykliczny graf stanów LangGraph z pętlami samonaprawczymi, lokalny akcelerator PyTorch na GPU RTX 2050 oraz streaming telemetrii i tracing w LangSmith.

---

## 1. Architektura Globalna i Przepływ End-to-End

Poniższy diagram przedstawia drogę danych od wejścia użytkownika lub pliku źródłowego, przez silniki przetwarzania i akcelerację sprzętową, aż po streaming i audyt jakości.

```mermaid
flowchart TD
    subgraph INGESTION["1. POTOK INGESTII & KNOWLEDGE LAKE"]
        SRC["Plik Źródłowy\n(PDF cyfrowy / Obraz JPG, PNG / TXT)"] --> NORM["DocumentNormalizer\n(RapidOCR + pdfplumber)"]
        NORM --> LAKE[("Markdown Knowledge Lake\ndata/knowledge_base/")]
        LAKE --> CHUNK["HierarchicalChunker\n(Parent: 1200 zn. | Child: 250 zn.)"]
        CHUNK --> VSTORE[("InMemoryVectorStore\n(Gemini Embeddings)")]
        CHUNK --> DOCSTORE[("Parent DocStore\n(Kompletne tabele i sekcje)")]
        VSTORE -.-> DISK_CACHE[("Trwały Dysk Cache\ndata/vector_cache/")]
        DOCSTORE -.-> DISK_CACHE
    end

    subgraph CLIENT_LAYER["2. WARSTWA KLIENTA & REST API"]
        USER(["Użytkownik / Przeglądarka Web"]) <-->|HTTP REST & Upload| FASTAPI["FastAPI Server (Port 8000)\n/api/v1/chat/stream\n/api/v1/documents"]
        FASTAPI -->|SSE Event Stream\nX-Accel-Buffering: no| UI["Web Workbench (HTML5 / Tailwind)\nMinimalistyczny Inspektor LLMOps"]
    end

    subgraph LANGGRAPH_ENGINE["3. CYKLICZNY GRAF STANÓW LANGGRAPH"]
        STATE[("GraphState\nThread ID + Messages Memory")]
        ROUTER{"Router Node\n(Gemini 3 Flash)"}
        RETRIEVE["Retrieve Node\n(Parent-Document Retrieval)"]
        RERANK["Local Rerank Node\n(PyTorch Cross-Encoder CUDA)"]
        GRADER{"Grade Documents Node\n(AsyncIO Parallel Evaluator)"}
        REWRITE["Rewrite Query Node\n(Adaptive Optimizer)"]
        GENERATE["Generate Node\n(Grounded Analytical Synthesis)"]
        GUARD{"Hallucination Guard Node\n(Groundedness & Relevance)"}
    end

    subgraph COMPUTE_LAYER["4. WARSTWA OBLICZENIOWA & TELEMETRIA"]
        GPU[/"NVIDIA GeForce RTX 2050 4GB\nPyTorch FP16 bge-reranker-base"/]
        GEMINI[/"Google Gemini API\ngemini-3-flash-preview & embeddings"/]
        SMITH[/"LangSmith Cloud Platform\nTracing & Latency Logs"/]
    end

    %% Połączenia między warstwami
    FASTAPI -->|Inicjalizacja zapytania| STATE
    STATE --> ROUTER
    ROUTER -->|Wymaga faktów z bazy| RETRIEVE
    ROUTER -->|Pytanie o fakty z pamięci| GENERATE

    RETRIEVE <-->|Similarity Search k=10| VSTORE
    VSTORE -.->|Automatyczna podmiana Child -> Parent| DOCSTORE
    RETRIEVE -->|Top 10 Parent Docs| RERANK

    RERANK <-->|Inferencja GPU FP16| GPU
    RERANK -->|Top 4 wyselekcjonowane chunki| GRADER

    GRADER <-->|Równoległa ewaluacja| GEMINI
    GRADER -->|Odrzucono wszystkie chunki| REWRITE
    REWRITE -->|Zoptymalizowane zapytanie| RETRIEVE
    GRADER -->|Zaakceptowano trafne chunki| GENERATE

    GENERATE <-->|Synteza analityczna| GEMINI
    GENERATE --> GUARD

    GUARD <-->|Podwójny audyt halucynacji| GEMINI
    GUARD -->|Wykryto halucynację| GENERATE
    GUARD -->|Zweryfikowano pomyślnie| FASTAPI

    LANGGRAPH_ENGINE -.->|Asynchroniczny tracing na żywo| SMITH
```

---

## 2. Maszyna Stanów LangGraph (State Machine Lifecycle)

System bazuje na cyklicznym grafie stanów ze zintegrowaną pamięcią konwersacyjną (`MemorySaver`) oraz dwiema pętlami samonaprawczymi:

```mermaid
stateDiagram-v2
    [*] --> Router: Zapytanie użytkownika + thread_id

    state Router {
        [*] --> AnalizaIntencji: Weryfikacja kontekstu konwersacji
        AnalizaIntencji --> DirectAnswer: Dopytanie o fakty z historii dialogu
        AnalizaIntencji --> RetrieveMode: Pytanie o nowe fakty i dokumenty
    }

    Router --> Generate: DirectAnswer (Ominięcie potoku RAG)
    Router --> Retrieve: RetrieveMode

    state Retrieve {
        [*] --> DenseSearch: Wyliczenie embeddingu pytania
        DenseSearch --> ParentSwap: Podmiana child chunków na pełny parent chunk
    }

    Retrieve --> LocalRerank: 10 Kandydatów (Parent Chunks)

    state LocalRerank {
        [*] --> PyTorchInference: Cross-Encoder na GPU RTX 2050 (FP16)
        PyTorchInference --> TopCandidates: Wybór Top-4 i obliczenie wag dopasowania
    }

    LocalRerank --> GradeDocuments: Top-4 Chunki + Rerank Scores

    state GradeDocuments {
        [*] --> AsyncGather: Równoległa ewaluacja każdego chunka przez LLM
        AsyncGather --> VerdictsLog: Zapis werdyktu (relevant/irrelevant) i uzasadnienia
    }

    GradeDocuments --> RewriteQuery: Żaden chunk nie jest relewantny (Noise Rejection)
    GradeDocuments --> Generate: Co najmniej 1 chunk merytorycznie poprawny

    state RewriteQuery {
        [*] --> ExtractKeywords: Wykrycie kluczowych fraz i sygnatur
        ExtractKeywords --> OptimizeQuery: Przeformułowanie pytania (Pętla #1)
    }

    RewriteQuery --> Retrieve: Nowe zoptymalizowane zapytanie do bazy

    state Generate {
        [*] --> ContextAssembly: Konstrukcja ścisłego promptu analitycznego
        ContextAssembly --> StrictSynthesis: Odpowiedź oparta w 100% na faktach ze źródeł
    }

    Generate --> HallucinationGuard: Wygenerowany tekst + Cytowane fragmenty

    state HallucinationGuard {
        [*] --> GroundednessCheck: Czy odpowiedź nie zawiera zmyślonych faktów?
        GroundednessCheck --> RelevanceCheck: Czy odpowiedź precyzyjnie odpowiada na pytanie?
    }

    HallucinationGuard --> Generate: Wykryto halucynację (Regeneracja max 1 raz)
    HallucinationGuard --> [*]: Odpowiedź ugruntowana i poprawna (Grounded & Useful)
```

---

## 3. Wzorzec Parent-Document Retrieval (Podwójna Ziarnistość)

W celu wyeliminowania rozmycia semantycznego oraz bezstratnego zachowania struktur tabel bilansowych (np. sprawozdań 10-Q), system dzieli wiedzę na dwa komplementarne poziomy:

```mermaid
graph TD
    DOC["Znormalizowany Plik Markdown (.md)\nnp. data/knowledge_base/finance/nvidia_q3_fy25_10q.md"] --> CHUNKER["HierarchicalChunker"]
    
    subgraph PARENT_LEVEL["Poziom Kontekstu (Parent DocStore)"]
        P1["Parent Chunk #1 (~1200 znaków)\nKompletna tabela bilansowa z nagłówkami"]
        P2["Parent Chunk #2 (~1200 znaków)\nOmówienie segmentu Data Center i architektury Blackwell"]
    end

    subgraph CHILD_LEVEL["Poziom Wyszukiwania (InMemoryVectorStore)"]
        C1_1["Child #1.1 (~250 zn.)\n'Przychody segmentu Data Center: $30.8 mld'"]
        C1_2["Child #1.2 (~250 zn.)\n'Marża brutto w Q3: 74.6%'"]
        C2_1["Child #2.1 (~250 zn.)\n'Dostawy układów Blackwell...'"]
    end

    CHUNKER --> P1
    CHUNKER --> P2
    P1 -->|Generuje haczyki| C1_1
    P1 -->|Generuje haczyki| C1_2
    P2 -->|Generuje haczyki| C2_1

    Q["Zapytanie: 'Ile wyniósł zysk netto NVIDIA w Q3 FY25?'"] -->|Podobieństwo Cosinusowe| C1_1
    C1_1 -.->|Pobranie parent_id| P1
    P1 ==>|Przekazanie pełnej tabeli| CTX["Pełny Kontekst Finansowy dla LLM"]
```

---

## 4. Schemat Strumienia Telemetrii SSE (Server-Sent Events)

FastAPI przesyła zdarzenia do przeglądarki w czasie rzeczywistym przez endpoint `/api/v1/chat/stream`:

| Zdarzenie SSE (`event`) | Węzeł (`node`) | Zawartość ładunku JSON (`data`) |
| :--- | :--- | :--- |
| `session` | - | Identyfikator sesji konwersacyjnej (`thread_id`) |
| `step` | `router` | Wybrany kierunek: `retrieve` lub `direct_answer`, czas trwania węzła w ms |
| `step` | `retrieve` | Lista surowych kandydatów z wektorówki (`index`, `filename`, `preview`, `is_table`) |
| `step` | `local_rerank` | Dokładne wagi Cross-Encodera (`score`), alokacja pamięci GPU VRAM w MB |
| `step` | `grade_documents` | Werdykty dla każdego chunku (`relevant`/`irrelevant`) wraz ze słownym uzasadnieniem LLM |
| `step` | `rewrite_query` | Informacja o pętli autokorekty: pierwotne zapytanie vs nowe zoptymalizowane |
| `step` | `generate` | Tekst odpowiedzi w Markdown oraz wykaz cytowanych źródeł (`citations`) |
| `step` | `hallucination_check` | Wyniki audytu: `grounded`/`not grounded` oraz `useful`/`not useful` |
| `complete` | - | Sygnał zakończenia generacji, całkowity czas przetwarzania (`total_time_ms`) |

---

## 5. Środowisko Sprzętowe i Optymalizacje

1. **Akceleracja GPU NVIDIA GeForce RTX 2050 (4096 MiB VRAM)**:
   - Dedykowany do inferencji Cross-Encodera `BAAI/bge-reranker-base`.
   - Obliczenia w półprecyzji `FP16` redukują narzut pamięci do poziomu ~350-480 MB VRAM, co gwarantuje stabilność pracy systemu.
2. **Współbieżność AsyncIO (HTTP/2 Multiplexing)**:
   - Węzły `grade_documents` oraz `hallucination_check` wykonują równoległe zapytania `ainvoke` przez `asyncio.gather`, co obniża czas odpowiedzi o ponad 50%.
3. **Trwałość Cache i Gotowość Produkcyjna**:
   - Stan indeksu wektorowego oraz słownik rodziców są trwale serializowane w `data/vector_cache/`, co umożliwia natychmiastowy start aplikacji bez powtórnego pobierania embeddingów.
