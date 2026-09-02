"""Węzeł transformacji zapytania (Rewrite Query Node) — samokorekta w pętli agentowej."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.state import GraphState


def rewrite_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu przeformułowujący zapytanie, gdy poprzednie wyszukiwanie nie przyniosło rezultatów.

    Optymalizuje słowa kluczowe i intencję zapytania pod kątem sprawozdań finansowych Big Tech.
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
        "Jesteś wyspecjalizowanym ekspertem wyszukiwania informacji w raportach finansowych (10-K, 10-Q).\n"
        "Użytkownik zadał pytanie, jednak wstępne przeszukanie bazy raportów NVIDIA i Alphabet nie przyniosło wystarczających danych.\n"
        "Twoim celem jest przeformułowanie i zoptymalizowanie zapytania, aby wydobyć kluczowe terminy finansowe "
        "(np. przychody, segment Data Center, Capex, marża brutto, H100, Blackwell, TPU, chmura).\n"
        "Zwróć WYŁĄCZNIE samo zoptymalizowane zapytanie, bez żadnych wstępów, cudzysłowów ani komentarzy."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Pierwotne pytanie: {original_question}\n"
                "Ostatnie (nieudane) zapytanie: {current_question}\n\n"
                "Zoptymalizowane nowe zapytanie do bazy wektorowej:",
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
    ).strip()

    return {
        "question": new_question,
        "retry_count": retry_count + 1,
    }
