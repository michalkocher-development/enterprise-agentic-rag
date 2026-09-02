"""Węzeł transformacji zapytania (Rewrite Query Node) — adaptacyjna samokorekta w pętli agentowej."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.state import GraphState


def rewrite_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu przeformułowujący zapytanie, gdy poprzednie wyszukiwanie nie przyniosło rezultatów.

    Działa w sposób uniwersalny dla wszystkich domen: finanse, prawo, edukacja, whitepapery AI.
    """
    settings = get_settings()
    question = state["question"]
    original_question = state.get("original_question", question)
    retry_count = state.get("retry_count", 0)

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.2,
        google_api_key=settings.google_api_key,
    )

    system_prompt = (
        "Jesteś ekspertem ds. optymalizacji zapytań wyszukiwawczych (Search Query Optimizer) w wielodomenowej bazie wiedzy.\n"
        "Wstępne przeszukanie bazy dokumentów dla zadanego pytania nie zwróciło relewantnych wyników lub dokumenty zostały odrzucone przez filtr.\n"
        "Twoim zadaniem jest przeformułować zapytanie, aby zmaksymalizować prawdopodobieństwo odnalezienia właściwych informacji:\n"
        "- Zidentyfikuj kluczowe pojęcia, słowa kluczowe, nazwy własne (np. numery dokumentów, sygnatury, tytuły, wskaźniki).\n"
        "- Usuń zbędne słowa gramatyczne i konwersacyjne ('jaki był', 'proszę podaj').\n"
        "- Zachowaj dokładne oznaczenia (np. '009', '10-Q', 'FY2025', 'EU AI Act').\n"
        "- Zwróć WYŁĄCZNIE samo zoptymalizowane zapytanie, bez żadnych wstępów, cudzysłowów ani komentarzy."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Pierwotne pytanie użytkownika: {original_question}\n"
                "Ostatnie (nieudane) zapytanie: {current_question}\n\n"
                "Zoptymalizowane zapytanie słów kluczowych do bazy wektorowej:",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    new_question = str(
        chain.invoke(
            {
                "original_question": original_question,
                "current_question": question,
            }
        )
    ).strip().strip('"').strip("'")

    return {
        "question": new_question,
        "retry_count": retry_count + 1,
        "rewrite_info": {
            "original_query": original_question,
            "failed_query": question,
            "new_query": new_question,
            "retry_number": retry_count + 1,
        },
    }
