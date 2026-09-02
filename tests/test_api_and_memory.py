"""Testy integracyjne dla pamięci konwersacyjnej (Adaptive RAG) oraz serwera FastAPI."""

import io
from fastapi.testclient import TestClient
import pytest

from src.api.server import app
from src.graph import app as agent_app
from src.state import GraphState


@pytest.fixture
def client():
    return TestClient(app)


def test_fastapi_health_endpoint(client):
    """Weryfikuje punkt kontrolny /api/v1/health i diagnostykę GPU/VRAM."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["device"] in ["cuda", "cpu"]
    assert "vector_cache_loaded" in data


def test_fastapi_openapi_docs(client):
    """Weryfikuje dostępność i strukturę dokumentacji OpenAPI (Swagger)."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Enterprise Agentic RAG Platform API"
    assert "/api/v1/chat" in schema["paths"]
    assert "/api/v1/chat/stream" in schema["paths"]
    assert "/api/v1/documents/upload" in schema["paths"]


def test_fastapi_document_upload_and_lake(client):
    """Weryfikuje endpoint /api/v1/documents/upload z automatyczną normalizacją do Markdown."""
    file_content = (
        "Artykuł 5 EU AI Act zakazuje systemów sztucznej inteligencji wykorzystujących techniki podprogowe "
        "oraz ocenę punktową obywateli (social scoring)."
    )
    fake_file = io.BytesIO(file_content.encode("utf-8"))

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("eu_ai_act_art5.txt", fake_file, "text/plain")},
        data={"domain": "legal"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["domain"] == "legal"
    assert data["filename"] == "eu_ai_act_art5.txt"
    assert "eu_ai_act_art5.md" in data["markdown_path"]


def test_conversational_memory_and_adaptive_router():
    """Weryfikuje pamięć wieloturową z thread_id i adaptacyjny routing w LangGraph."""
    thread_id = "test-memory-session-42"
    config = {"configurable": {"thread_id": thread_id}}

    # Tura 1: Pytanie wymagające sięgnięcia do bazy dokumentów (RAG)
    q1 = "Ile wynosił zysk netto (Net Income) spółki NVIDIA w Q3 FY2025 według oficjalnego raportu 10-Q?"
    state_1: GraphState = {
        "question": q1,
        "original_question": q1,
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

    res_1 = agent_app.invoke(state_1, config=config)
    ans_1 = res_1.get("generation", "")
    assert ("19 309" in ans_1 or "19,309" in ans_1 or "19.309" in ans_1)
    assert len(res_1.get("messages", [])) >= 2

    # Tura 2: Dopytanie nawiązujące do pierwszej tury (Router powinien obsłużyć to w oparciu o pamięć)
    q2 = "A o jakiej spółce technologicznej przed chwilą rozmawialiśmy?"
    state_2: GraphState = {
        "question": q2,
        "original_question": q2,
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

    res_2 = agent_app.invoke(state_2, config=config)
    ans_2 = res_2.get("generation", "")
    # Sprawdzamy czy model pamięta, że rozmowa dotyczyła NVIDIA
    assert "NVIDIA" in ans_2 or "Nvidia" in ans_2


def test_fastapi_chat_stream_sse(client):
    """Weryfikuje streaming zdarzeń Server-Sent Events (SSE)."""
    payload = {
        "question": "Podaj jedno zdanie o EU AI Act.",
        "thread_id": "test-sse-thread-1",
    }
    response = client.post("/api/v1/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    content = response.text
    assert "event: session" in content
    assert "event: step" in content
    assert "event: complete" in content
