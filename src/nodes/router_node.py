"""Węzeł adaptacyjnego Routera (Adaptive RAG) decydujący o potrzebie sięgnięcia do bazy dokumentów."""

from typing import Any, Dict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import get_settings
from src.state import GraphState


class RouteDecision(BaseModel):
    """Decyzja routingu zapytania użytkownika."""

    route: Literal["retrieve", "direct_answer"] = Field(
        description="Wybierz 'direct_answer' jeśli pytanie jest powitaniem, nawiązaniem do faktów już podanych w historii rozmowy, "
        "lub pytaniem ogólnym. Wybierz 'retrieve' jeśli pytanie wymaga odnalezienia konkretnych faktów w dokumentach zewnętrznych."
    )
    reasoning: str = Field(description="Zwięzłe uzasadnienie decyzji (1 zdanie).")


ROUTER_PROMPT = """Jesteś ekspertem ds. routingu zapytań w systemie Agentic RAG.
Twoim zadaniem jest ocenić, czy nowe zapytanie użytkownika wymaga przeszukania zewnętrznej bazy dokumentów ('retrieve'),
czy może zostać obsłużone bezpośrednio ('direct_answer') na podstawie:
1. Pamięci dotychczasowego dialogu (jeśli użytkownik dopytuje o szczegóły faktu, który już pojawił się we wcześniejszych wiadomościach asystenta),
2. Zwykłej konwersacji (powitanie, podziękowanie, prośba o podsumowanie poprzedniej wypowiedzi).

ZASADY:
- Jeśli użytkownik pyta o nowe dane liczbowe, konkretne artykuły prawne (np. EU AI Act), sprawozdania finansowe (np. NVIDIA, TSMC) lub nieznany dokument -> WYBIERZ 'retrieve'.
- Jeśli odpowiedź na pytanie padła już w historii rozmowy (np. Użytkownik: 'A ile wynosiła waga tego?' odnosząc się do wymienionego wcześniej dyplomu) -> WYBIERZ 'direct_answer'.
- Jeśli użytkownik pisze 'Cześć', 'Kim jesteś?', 'Dzięki za pomoc' -> WYBIERZ 'direct_answer'.
"""


def route_question_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł analizujący zapytanie i historię wiadomości pod kątem konieczności uruchomienia RAG."""
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )
    structured_router = llm.with_structured_output(RouteDecision)

    question = state.get("question", "")
    messages = state.get("messages", [])

    history_summary = []
    for msg in messages[-6:]:  # Ostatnie 3 tury
        sender = "Użytkownik" if isinstance(msg, HumanMessage) else "Asystent"
        history_summary.append(f"{sender}: {msg.content[:200]}")

    formatted_history = "\n".join(history_summary) if history_summary else "(Brak wcześniejszej historii)"

    user_prompt = f"""HISTORIA ROZMOWY:
{formatted_history}

NOWE ZAPYTANIE:
{question}
"""

    try:
        decision: RouteDecision = structured_router.invoke(
            [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=user_prompt)]
        )
        route = decision.route
    except Exception as e:
        print(f"Błąd podczas ewaluacji routingu: {e}. Domyślny fallback: 'retrieve'.")
        route = "retrieve"

    return {"route": route}
