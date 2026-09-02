"""Główna aplikacja FastAPI: punkty końcowe REST, streaming SSE oraz automatyczny Swagger UI."""

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document

from src.api.replays import REPLAY_SCENARIOS, stream_replay_events
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentItem,
    HealthResponse,
    IngestResponse,
    ReplayScenarioItem,
    StepLog,
)
from src.config import get_settings
from src.graph import app as agent_app
from src.ingestion.document_normalizer import DocumentNormalizer
from src.reranker.local_reranker import get_reranker
from src.retriever.vector_store import get_vector_store
from src.state import GraphState

_GPU_READY = False
_MODEL_WARMED_UP = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciepły start (Warmup): rozgrzewka bazy wektorowej oraz Cross-Encodera na GPU."""
    global _GPU_READY, _MODEL_WARMED_UP
    try:
        # 0. Wczytanie konfiguracji i synchronizacja os.environ dla LangSmith
        settings = get_settings()

        # 1. Inicjalizacja bazy wektorowej
        get_vector_store()

        # 2. Warmup Cross-Encodera (1 próbny forward pass)
        reranker = get_reranker()
        reranker._load_model_if_needed()
        reranker.rank("warmup", [Document(page_content="warmup candidate")], top_k=1)

        _GPU_READY = torch.cuda.is_available()
        _MODEL_WARMED_UP = True
        langsmith_status = "aktywny" if (settings.langchain_api_key and settings.langchain_tracing_v2) else "wyłączony"
        print(f"[Lifespan] Baza wektorowa i model Cross-Encoder rozgrzane. Tracing LangSmith: {langsmith_status}.")
    except Exception as exc:
        print(f"[Lifespan] Ostrzeżenie podczas warmup: {exc}")
        _GPU_READY = torch.cuda.is_available()
        _MODEL_WARMED_UP = False

    yield


app = FastAPI(
    title="Enterprise Agentic RAG Platform API",
    version="2.0.0",
    description=(
        "Produkcyjny interfejs REST API dla wielodomenowej platformy Agentic Document Intelligence.\n\n"
        "**Kluczowe możliwości**:\n"
        "- **Adaptive RAG z pamięcią**: Automatyczny routing zapytań (pamięć wieloturowa vs RAG w dokumentacji),\n"
        "- **Dwufazowy potok Ingestion**: Konwersja chaotycznych PDF, skanów OCR i TXT do Markdown Knowledge Lake,\n"
        "- **Akceleracja GPU**: Reranking lokalny Cross-Encoder w PyTorch (FP16 na CUDA),\n"
        "- **Streaming SSE**: Śledzenie wykonania kolejnych węzłów grafu LangGraph w czasie rzeczywistym."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System & Diagnostics"])
def health_check() -> HealthResponse:
    """Zwraca stan zdrowia serwisu, wykorzystywany sprzęt oraz telemetrię pamięci VRAM i LangSmith."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vram_mb = (
        torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else None
    )

    settings = get_settings()
    langsmith_on = bool(
        os.environ.get("LANGCHAIN_TRACING_V2") == "true" and os.environ.get("LANGCHAIN_API_KEY")
    )

    vs = get_vector_store()
    kb_dir = Path("data/knowledge_base")
    kb_count = len(list(kb_dir.rglob("*.md"))) if kb_dir.exists() else 0

    return HealthResponse(
        status="healthy",
        device=device,
        gpu_ready=_GPU_READY or (device == "cuda"),
        model_warmed_up=_MODEL_WARMED_UP,
        langsmith_active=langsmith_on,
        gpu_name=gpu_name,
        vram_allocated_mb=vram_mb,
        vector_cache_loaded=vs._is_populated,
        knowledge_base_documents=kb_count,
    )


@app.get("/api/v1/documents", response_model=List[DocumentItem], tags=["Knowledge Lake"])
def list_knowledge_documents() -> List[DocumentItem]:
    """Zwraca listę wszystkich ustrukturyzowanych dokumentów Markdown znajdujących się w Knowledge Lake."""
    kb_dir = Path("data/knowledge_base")
    if not kb_dir.exists():
        return []

    items: List[DocumentItem] = []
    for file_path in sorted(kb_dir.rglob("*.md")):
        rel_path = str(file_path.relative_to(kb_dir))
        domain = file_path.parent.name
        title = file_path.stem.replace("_", " ").title()
        size = file_path.stat().st_size

        # Odczyt nagłówka YAML pod kątem metadanych
        tables_count = 0
        ocr_used = False
        try:
            content = file_path.read_text(encoding="utf-8")
            if "tables_count:" in content:
                for line in content.splitlines()[:12]:
                    if line.startswith("tables_count:"):
                        tables_count = int(line.split(":")[-1].strip())
                    elif line.startswith("ocr_used:"):
                        ocr_used = "true" in line.lower()
                    elif line.startswith("title:"):
                        title = line.split(":", 1)[-1].strip().strip('"')
        except Exception:
            pass

        items.append(
            DocumentItem(
                filename=file_path.name,
                domain=domain,
                title=title,
                relative_path=rel_path,
                size_bytes=size,
                tables_count=tables_count,
                ocr_used=ocr_used,
            )
        )
    return items


@app.get("/api/v1/documents/{domain}/{filename}", tags=["Knowledge Lake"])
def get_document_content(domain: str, filename: str):
    """Zwraca pełną treść i metadane pojedynczego dokumentu z repozytorium Knowledge Lake."""
    file_path = Path("data/knowledge_base") / domain / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Dokument nie został odnaleziony w bazie wiedzy.")
    content = file_path.read_text(encoding="utf-8")
    return {
        "filename": filename,
        "domain": domain,
        "size_bytes": file_path.stat().st_size,
        "content": content,
    }



@app.post("/api/v1/documents/upload", response_model=IngestResponse, tags=["Knowledge Lake"])
async def upload_document(
    file: UploadFile = File(...),
    domain: str = Form(default="general"),
) -> IngestResponse:
    """Odbiera plik (PDF, JPG/PNG, TXT), przetwarza go przez silnik OCR/tabel i zapisuje do Markdown Lake."""
    normalizer = DocumentNormalizer()
    suffix = Path(file.filename).suffix.lower()

    if suffix not in [".pdf", ".png", ".jpg", ".jpeg", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Nieobsługiwany format pliku: {suffix}. Dozwolone: .pdf, .png, .jpg, .jpeg, .txt",
        )

    # Zapis tymczasowy pliku
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        norm_doc = normalizer.normalize_file(tmp_path, domain=domain)
        # Zachowujemy oryginalną nazwę pliku w tytule
        norm_doc.title = Path(file.filename).stem.replace("_", " ").title()
        norm_doc.metadata["source_file"] = file.filename

        saved_path = normalizer.save_to_knowledge_lake(norm_doc)

        # Dynamiczne zasilenie bazy wektorowej nowym dokumentem
        vs = get_vector_store()
        vs.add_markdown_file(saved_path, domain=domain)

        return IngestResponse(
            success=True,
            filename=file.filename,
            domain=domain,
            title=norm_doc.title,
            markdown_path=str(saved_path),
            pages_count=norm_doc.metadata.get("pages_count", 1),
            tables_count=len(norm_doc.tables),
            ocr_used=norm_doc.metadata.get("ocr_used", False),
            message="Plik został pomyślnie znormalizowany i zindeksowany w bazie wiedzy.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd przetwarzania pliku: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Agentic Core"])
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Synchroniczne wywołanie agenta z pamięcią wieloturową (thread_id) i audytem kroków."""
    t0 = time.time()
    thread_id = request.thread_id or f"session-{os.urandom(4).hex()}"

    initial_state: GraphState = {
        "question": request.question,
        "original_question": request.question,
        "documents": [],
        "rerank_scores": [],
        "generation": None,
        "retry_count": 0,
        "regeneration_count": 0,
        "retry_limit": 2,
        "regeneration_limit": 1,
        "lang": request.lang or "pl",
        "hallucination_grade": None,
        "answer_grade": None,
        "web_search_needed": False,
        "route": None,
        "messages": [],
    }

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"domain": request.domain},
    }

    steps_log: List[StepLog] = []
    citations: List[Citation] = []
    final_generation = ""
    route_taken = "retrieve"
    hallucination_grade = None

    # Strumieniowe śledzenie węzłów
    for event in agent_app.stream(initial_state, config=config):
        for node_name, state_update in event.items():
            if node_name == "router":
                route_taken = state_update.get("route", "retrieve")
                desc = (
                    "Pominięto RAG — odpowiedź zostanie udzielona bezpośrednio z pamięci dialogu."
                    if route_taken == "direct_answer"
                    else "Zidentyfikowano potrzebę sięgnięcia do zewnętrznej bazy dokumentacji."
                )
                steps_log.append(StepLog(node="router", description=desc, data={"route": route_taken}))

            elif node_name == "retrieve":
                docs = state_update.get("documents", [])
                steps_log.append(
                    StepLog(
                        node="retrieve",
                        description=f"Pobrano {len(docs)} kandydackich Parent Chunks z bazy wiedzy.",
                        data={"candidate_count": len(docs)},
                    )
                )

            elif node_name == "local_rerank":
                scores = state_update.get("rerank_scores", [])
                docs = state_update.get("documents", [])
                steps_log.append(
                    StepLog(
                        node="local_rerank",
                        description=f"Cross-Encoder na GPU przeliczył wagi relewantności dla {len(docs)} fragmentów.",
                        data={"top_scores": [round(s, 3) for s in scores[:3]]},
                    )
                )

            elif node_name == "grade_documents":
                docs = state_update.get("documents", [])
                needed = state_update.get("web_search_needed", False)
                desc = "Odrzucono nieadekwatne chunki — zapytanie wymaga przeredagowania." if needed else f"Asynchroniczny LLM Grader zaakceptował {len(docs)} fragmentów."
                steps_log.append(StepLog(node="grade_documents", description=desc, data={"accepted_count": len(docs)}))

            elif node_name == "rewrite_query":
                new_q = state_update.get("question", "")
                retries = state_update.get("retry_count", 0)
                steps_log.append(
                    StepLog(
                        node="rewrite_query",
                        description=f"Autokorekta zapytania #{retries}.",
                        data={"new_question": new_q},
                    )
                )

            elif node_name == "generate":
                final_generation = state_update.get("generation", "")
                docs = state_update.get("documents", [])
                steps_log.append(StepLog(node="generate", description="Gemini 3 Flash wygenerował odpowiedź."))
                for idx, d in enumerate(docs):
                    citations.append(
                        Citation(
                            chunk_id=d.metadata.get("chunk_id", f"{d.metadata.get('filename', 'doc')}#p{idx}"),
                            source=d.metadata.get("source", ""),
                            filename=d.metadata.get("filename", ""),
                            company_or_domain=d.metadata.get("company", d.metadata.get("domain", "general")),
                            content_snippet=d.page_content[:300],
                            is_table=d.metadata.get("is_table", False),
                        )
                    )

            elif node_name == "hallucination_check":
                hallucination_grade = state_update.get("hallucination_grade", "grounded")
                answer_grade = state_update.get("answer_grade", "useful")
                steps_log.append(
                    StepLog(
                        node="hallucination_check",
                        description=f"Audyt ugruntowania: {hallucination_grade.upper()}, celność: {answer_grade.upper()}.",
                        data={"hallucination": hallucination_grade, "relevance": answer_grade},
                    )
                )

    latency_ms = round((time.time() - t0) * 1000, 2)

    return ChatResponse(
        answer=final_generation,
        thread_id=thread_id,
        route_taken=route_taken,
        steps_log=steps_log,
        citations=citations,
        hallucination_grade=hallucination_grade,
        latency_ms=latency_ms,
    )


GRAPH_NODES = {
    "router",
    "retrieve",
    "local_rerank",
    "grade_documents",
    "rewrite_query",
    "generate",
    "hallucination_check",
}


def _extract_delta(chunk: Any) -> str:
    """Ekstrahuje przyrost tekstu (delta token) z obiektu strumienia LLM."""
    if not chunk:
        return ""
    if isinstance(chunk, str):
        return chunk
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return "".join(parts)
    return str(content) if content else ""


def _build_step_payload(
    node: str,
    output: Dict[str, Any],
    duration_ms: float,
    total_elapsed_ms: float,
    iteration: int,
    tokens_accumulated: List[str],
) -> Dict[str, Any]:
    """Konstruuje znormalizowany i wzbogacony ładunek zdarzenia step dla UI."""
    payload: Dict[str, Any] = {
        "node": node,
        "duration_ms": duration_ms,
        "latency_ms": duration_ms,
        "total_ms": total_elapsed_ms,
        "iteration": iteration,
        "retry_limit": 2,
        "regeneration_limit": 1,
        "route": output.get("route"),
        "documents_count": len(output.get("documents", [])),
        "rerank_scores": [round(s, 3) for s in output.get("rerank_scores", [])],
        "generation": output.get("generation"),
        "hallucination_grade": output.get("hallucination_grade"),
        "answer_grade": output.get("answer_grade"),
        "web_search_needed": output.get("web_search_needed", False),
    }

    if node == "retrieve":
        docs = output.get("documents", [])
        payload["candidates"] = [
            {
                "index": idx + 1,
                "chunk_id": d.metadata.get("chunk_id", f"{Path(d.metadata.get('filename', 'doc')).stem}#p{idx}"),
                "filename": d.metadata.get("filename", "unknown"),
                "company": d.metadata.get("company", d.metadata.get("domain", "general")),
                "is_table": d.metadata.get("is_table", False),
                "preview": d.page_content[:300],
                "full_text": d.page_content,
            }
            for idx, d in enumerate(docs)
        ]

    elif node == "local_rerank":
        ranked = output.get("ranked_candidates", [])
        payload["gpu_device"] = (
            "NVIDIA GeForce RTX 2050 (FP16)" if torch.cuda.is_available() else "CPU Mode"
        )
        payload["vram_mb"] = (
            round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            if torch.cuda.is_available()
            else 0.0
        )
        payload["ranked"] = ranked
        payload["ranked_candidates"] = ranked

    elif node == "grade_documents":
        payload["graded_verdicts"] = output.get("graded_verdicts", [])
        payload["accepted_count"] = len(output.get("documents", []))
        payload["rejected_count"] = len(output.get("graded_verdicts", [])) - len(output.get("documents", []))
        payload["web_search_needed"] = output.get("web_search_needed", False)

    elif node == "rewrite_query":
        payload["rewrite_info"] = output.get("rewrite_info", {})
        payload["new_question"] = output.get("question", "")
        payload["retry_count"] = output.get("retry_count", iteration)

    elif node == "generate":
        docs = output.get("documents", [])
        payload["generation"] = output.get("generation") or "".join(tokens_accumulated)
        payload["citations"] = [
            {
                "chunk_id": d.metadata.get("chunk_id", f"{Path(d.metadata.get('filename', 'doc')).stem}#p{idx}"),
                "filename": d.metadata.get("filename", "unknown"),
                "company": d.metadata.get("company", "general"),
                "is_table": d.metadata.get("is_table", False),
                "snippet": d.page_content[:400],
                "full_text": d.page_content,
            }
            for idx, d in enumerate(docs)
        ]

    return payload


async def generate_chat_sse(
    question: str,
    lang: str = "pl",
    thread_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Centralny generator Server-Sent Events (SSE) dla żądań GET i POST ze streamingiem tokenów i keepalive."""
    thread_id = thread_id or f"session-{os.urandom(4).hex()}"
    t0 = time.time()

    initial_state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "rerank_scores": [],
        "generation": None,
        "retry_count": 0,
        "regeneration_count": 0,
        "retry_limit": 2,
        "regeneration_limit": 1,
        "lang": lang,
        "hallucination_grade": None,
        "answer_grade": None,
        "web_search_needed": False,
        "route": None,
        "messages": [],
    }

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"domain": domain},
    }

    queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()

    await queue.put(f"event: session\ndata: {json.dumps({'thread_id': thread_id})}\n\n")

    async def keepalive_worker():
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                if not stop_event.is_set():
                    await queue.put(": keepalive\n\n")

    async def workflow_worker():
        last_node = "START"
        active_node = None
        iteration = 0
        node_start_time: Dict[str, float] = {}
        tokens_accumulated: List[str] = []

        try:
            async for event in agent_app.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]
                meta = event.get("metadata", {})
                node = meta.get("langgraph_node")

                if kind == "on_chain_start" and node in GRAPH_NODES and node != active_node:
                    edge_from = last_node
                    active_node = node
                    node_start_time[node] = time.time()
                    start_payload = {
                        "node": node,
                        "edge_from": edge_from,
                        "iteration": iteration,
                    }
                    await queue.put(
                        f"event: node_start\ndata: {json.dumps(start_payload, ensure_ascii=False)}\n\n"
                    )

                elif kind == "on_chat_model_stream" and active_node == "generate":
                    chunk = event.get("data", {}).get("chunk")
                    delta = _extract_delta(chunk)
                    if delta:
                        tokens_accumulated.append(delta)
                        await queue.put(
                            f"event: token\ndata: {json.dumps({'node': 'generate', 'delta': delta}, ensure_ascii=False)}\n\n"
                        )

                elif kind == "on_chain_end" and node in GRAPH_NODES:
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict):
                        now = time.time()
                        start_t = node_start_time.get(node, now)
                        duration_ms = round((now - start_t) * 1000, 1)
                        total_elapsed_ms = round((now - t0) * 1000, 1)
                        last_node = node
                        active_node = None

                        if node == "rewrite_query":
                            iteration = output.get("retry_count", iteration + 1)

                        step_payload = _build_step_payload(
                            node=node,
                            output=output,
                            duration_ms=duration_ms,
                            total_elapsed_ms=total_elapsed_ms,
                            iteration=iteration,
                            tokens_accumulated=tokens_accumulated,
                        )
                        await queue.put(
                            f"event: step\ndata: {json.dumps(step_payload, ensure_ascii=False)}\n\n"
                        )

            total_time_ms = round((time.time() - t0) * 1000, 1)
            await queue.put(
                f"event: complete\ndata: {json.dumps({'status': 'done', 'thread_id': thread_id, 'total_time_ms': total_time_ms})}\n\n"
            )
        except Exception as exc:
            err_payload = {
                "node": active_node or "system",
                "code": "rate_limit" if ("429" in str(exc) or "ResourceExhausted" in str(exc)) else "execution_error",
                "message": str(exc),
            }
            await queue.put(f"event: error\ndata: {json.dumps(err_payload, ensure_ascii=False)}\n\n")
        finally:
            stop_event.set()
            await queue.put(None)

    workflow_task = asyncio.create_task(workflow_worker())
    keepalive_task = asyncio.create_task(keepalive_worker())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        workflow_task.cancel()
        keepalive_task.cancel()


@app.get("/api/v1/chat/stream", tags=["Agentic Core"])
async def chat_stream_get_endpoint(
    q: str = Query(..., description="Treść zapytania użytkownika"),
    lang: str = Query(default="pl", description="Język odpowiedzi ('pl' lub 'en')"),
    thread_id: Optional[str] = Query(default=None, description="Identyfikator wątku sesji"),
    domain: Optional[str] = Query(default=None, description="Opcjonalna domena"),
):
    """Natywny punkt końcowy SSE dla przeglądarkowego EventSource (metoda HTTP GET)."""
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        generate_chat_sse(question=q, lang=lang, thread_id=thread_id, domain=domain),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/api/v1/chat/stream", tags=["Agentic Core"])
async def chat_stream_post_endpoint(request: ChatRequest):
    """Punkt końcowy SSE dla klientów HTTP wysyłających ciało żądania w formacie JSON (POST)."""
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        generate_chat_sse(
            question=request.question,
            lang=request.lang or "pl",
            thread_id=request.thread_id,
            domain=request.domain,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get("/api/v1/replays", response_model=List[ReplayScenarioItem], tags=["Portfolio Demo Replay"])
def list_replay_scenarios() -> List[ReplayScenarioItem]:
    """Zwraca listę gotowych scenariuszy demonstracyjnych do bezkosztowego odtwarzania w UI."""
    return [
        ReplayScenarioItem(
            id=s["id"],
            title=s["title"],
            description=s["description"],
            query=s["query"],
            has_self_correction=s["has_self_correction"],
            estimated_duration_s=s["estimated_duration_s"],
        )
        for s in REPLAY_SCENARIOS.values()
    ]


@app.get("/api/v1/replay/{run_id}", tags=["Portfolio Demo Replay"])
async def replay_scenario_endpoint(
    run_id: str,
    tempo: float = Query(default=1.0, description="Mnożnik tempa odtwarzania (np. 1.0, 2.0, 4.0)"),
):
    """Odtwarza wzorcowy przebieg zdarzeń SSE z wybranym tempem demo (1x, 2x, 4x)."""
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        stream_replay_events(scenario_id=run_id, tempo=tempo),
        media_type="text/event-stream",
        headers=headers,
    )


# Montowanie statycznego frontendu na ścieżce głównej / (po zarejestrowaniu endpointów API)
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

