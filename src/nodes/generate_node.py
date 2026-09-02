"""Węzeł generowania (Generate Node) — synteza ze zweryfikowanego kontekstu lub pamięci dialogu."""

from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.state import GraphState


def _clean_generation_text(raw_content: Any) -> str:
    """Ekstrahuje czysty tekst z odpowiedzi LLM, eliminując metadane sygnatur i zagnieżdżone słowniki."""
    if isinstance(raw_content, str):
        return raw_content.strip()
    elif isinstance(raw_content, list):
        text_parts = []
        for item in raw_content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
                elif "text" in item and not item.get("extras"):
                    text_parts.append(str(item["text"]))
        return "\n".join(text_parts).strip()
    return str(raw_content).strip()


def generate_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu syntetyzujący odpowiedź:
    - w trybie 'direct_answer' odpowiada w oparciu o pamięć dotychczasowej rozmowy,
    - w trybie RAG precyzyjnie cytuje fakty ze zweryfikowanych dokumentów.
    """
    settings = get_settings()
    question = state["original_question"]
    documents = state.get("documents", [])
    route = state.get("route", "retrieve")
    messages = state.get("messages", [])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
        google_api_key=settings.google_api_key,
    )

    reg_count = state.get("regeneration_count", 0)
    if state.get("generation"):
        reg_count += 1

    if route == "direct_answer":
        # Odpowiedź konwersacyjna z pamięci dialogu
        system_prompt = (
            "Jesteś asystentem analitycznym platformy Enterprise Agentic RAG.\n"
            "Odpowiadasz na pytanie użytkownika na podstawie historii dotychczasowej rozmowy.\n"
            "Jeśli użytkownik dopytuje o fakty podane wcześniej (np. oceny, liczby, daty), precyzyjnie je przytocz.\n"
            "Odpowiadaj w języku polskim, zwięźle i naturalnie w czystym formacie Markdown."
        )
        chat_messages = [SystemMessage(content=system_prompt)]
        chat_messages.extend(messages[-6:])
        chat_messages.append(HumanMessage(content=question))

        raw_res = llm.invoke(chat_messages).content
        generation = _clean_generation_text(raw_res)

        return {
            "generation": generation,
            "regeneration_count": reg_count,
            "messages": [HumanMessage(content=question), AIMessage(content=generation)],
            "hallucination_grade": "grounded",
            "answer_grade": "useful",
        }

    else:
        # Odpowiedź ugruntowana w pobranych dokumentach (RAG)
        context_str = "\n\n---\n\n".join(
            [f"[Dokument: {doc.metadata.get('filename', 'źródło')}]\n{doc.page_content}" for doc in documents]
        )

        system_prompt = (
            "Jesteś precyzyjnym analitykiem technologicznym i biznesowym platformy Enterprise Agentic RAG.\n"
            "Odpowiadasz na pytania użytkownika WYŁĄCZNIE na podstawie dostarczonego poniżej kontekstu z bazy wiedzy.\n"
            "Zasady bezwzględne:\n"
            "1. Podawaj dokładne liczby, artykuły prawne, parametry techniczne i wskaźniki dokładnie tak, jak w źródłach.\n"
            "2. Jeśli kontekst nie zawiera odpowiedzi, wprost zaznacz brak tych danych w dokumentacji.\n"
            "3. Nie twórz domysłów ani nie wprowadzaj faktów spoza dostarczonego tekstu.\n"
            "4. Odpowiadaj w języku polskim, profesjonalnie i formatuj tabele oraz wyliczenia w czytelnym Markdown."
        )

        user_content = f"Kontekst z bazy wiedzy:\n{context_str}\n\nPytanie: {question}\n\nOdpowiedź analityczna:"
        chat_messages = [SystemMessage(content=system_prompt)]
        chat_messages.extend(messages[-4:])
        chat_messages.append(HumanMessage(content=user_content))

        raw_res = llm.invoke(chat_messages).content
        generation = _clean_generation_text(raw_res)

        return {
            "generation": generation,
            "documents": documents,
            "regeneration_count": reg_count,
            "messages": [HumanMessage(content=question), AIMessage(content=generation)],
        }
