"""Główna aplikacja FastAPI: punkty końcowe REST, streaming SSE oraz automatyczny Swagger UI."""

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentItem,
    HealthResponse,
    IngestResponse,
    StepLog,
)
from src.config import get_settings
from src.graph import app as agent_app
from src.ingestion.document_normalizer import DocumentNormalizer
from src.retriever.vector_store import get_vector_store
from src.state import GraphState


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
    """Zwraca stan zdrowia serwisu, wykorzystywany sprzęt oraz telemetrię pamięci VRAM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vram_mb = (
        torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else None
    )

    vs = get_vector_store()
    kb_dir = Path("data/knowledge_base")
    kb_count = len(list(kb_dir.rglob("*.md"))) if kb_dir.exists() else 0

    return HealthResponse(
        status="healthy",
        device=device,
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

        # Zasilenie bazy wektorowej nowym plikiem
        vs = get_vector_store()
        # Wczytanie do wektorówki
        vs.load_from_directory()

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
                for d in docs:
                    citations.append(
                        Citation(
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


@app.post("/api/v1/chat/stream", tags=["Agentic Core"])
async def chat_stream_endpoint(request: ChatRequest):
    """Strumieniowanie Server-Sent Events (SSE): przesyła zdarzenia węzłów, szczegółowe metryki GPU i tokeny w czasie rzeczywistym."""
    thread_id = request.thread_id or f"session-{os.urandom(4).hex()}"

    initial_state: GraphState = {
        "question": request.question,
        "original_question": request.question,
        "documents": [],
        "rerank_scores": [],
        "generation": None,
        "retry_count": 0,
        "regeneration_count": 0,
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

    async def event_generator() -> AsyncGenerator[str, None]:
        t0 = time.time()
        last_step_time = t0

        yield f"event: session\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
        await asyncio.sleep(0.01)

        async for event in agent_app.astream(initial_state, config=config):
            now = time.time()
            step_latency_ms = round((now - last_step_time) * 1000, 1)
            total_elapsed_ms = round((now - t0) * 1000, 1)
            last_step_time = now

            for node_name, state_update in event.items():
                payload = {
                    "node": node_name,
                    "latency_ms": step_latency_ms,
                    "total_ms": total_elapsed_ms,
                    "route": state_update.get("route"),
                    "documents_count": len(state_update.get("documents", [])),
                    "rerank_scores": [round(s, 3) for s in state_update.get("rerank_scores", [])],
                    "generation": state_update.get("generation"),
                    "hallucination_grade": state_update.get("hallucination_grade"),
                    "answer_grade": state_update.get("answer_grade"),
                    "web_search_needed": state_update.get("web_search_needed", False),
                }

                # Pełna telemetria kandydatów i tabel
                if node_name == "retrieve":
                    docs = state_update.get("documents", [])
                    payload["candidates"] = [
                        {
                            "index": idx + 1,
                            "filename": d.metadata.get("filename", "unknown"),
                            "company": d.metadata.get("company", d.metadata.get("domain", "general")),
                            "is_table": d.metadata.get("is_table", False),
                            "preview": d.page_content[:300],
                            "full_text": d.page_content,
                        }
                        for idx, d in enumerate(docs[:6])
                    ]

                elif node_name == "local_rerank":
                    docs = state_update.get("documents", [])
                    scores = state_update.get("rerank_scores", [])
                    payload["gpu_device"] = (
                        "NVIDIA GeForce RTX 2050 (FP16)" if torch.cuda.is_available() else "CPU Mode"
                    )
                    payload["vram_mb"] = (
                        round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    payload["ranked_candidates"] = [
                        {
                            "rank": idx + 1,
                            "filename": d.metadata.get("filename", "unknown"),
                            "score": round(scores[idx], 3) if idx < len(scores) else 0.0,
                            "is_table": d.metadata.get("is_table", False),
                            "preview": d.page_content[:250],
                            "full_text": d.page_content,
                        }
                        for idx, d in enumerate(docs[:6])
                    ]

                elif node_name == "generate":
                    docs = state_update.get("documents", [])
                    payload["citations"] = [
                        {
                            "filename": d.metadata.get("filename", "unknown"),
                            "company": d.metadata.get("company", "general"),
                            "is_table": d.metadata.get("is_table", False),
                            "snippet": d.page_content[:400],
                            "full_text": d.page_content,
                        }
                        for d in docs[:4]
                    ]

                yield f"event: step\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

        total_time_ms = round((time.time() - t0) * 1000, 1)
        yield f"event: complete\ndata: {json.dumps({'status': 'done', 'thread_id': thread_id, 'total_time_ms': total_time_ms})}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


# Montowanie statycznego frontendu na ścieżce głównej / (po zarejestrowaniu endpointów API)
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

