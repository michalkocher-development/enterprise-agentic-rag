"""Węzeł sprawdzania halucynacji (Hallucination Node) — równoległa weryfikacja ugruntowania i celności (asyncio)."""

import asyncio
from typing import Any, Dict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import get_settings
from src.state import GraphState
from src.utils.async_runner import run_async


class GradeHallucinations(BaseModel):
    """Ocena czy generacja jest ugruntowana w faktach (brak halucynacji)."""

    binary_score: Literal["yes", "no"] = Field(
        description="Czy odpowiedź jest poparta faktami z dokumentów: 'yes' (ugruntowana) lub 'no' (zawiera halucynacje)."
    )


class GradeAnswer(BaseModel):
    """Ocena czy generacja odpowiada na zadane pytanie użytkownika."""

    binary_score: Literal["yes", "no"] = Field(
        description="Czy odpowiedź bezpośrednio rozwiązuje pytanie: 'yes' (adekwatna) lub 'no' (omija pytanie)."
    )


def hallucination_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu weryfikujący jakość wygenerowanej odpowiedzi.
    
    Wykonuje równolegle dwa testy przez asyncio.gather:
    1. Groundedness (sprawdzenie czy fakty pochodzą ze sprawozdań).
    2. Answer Relevance (sprawdzenie czy odpowiedź odpowiada na pytanie).
    """
    settings = get_settings()
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    question = state.get("original_question", "")

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
        google_api_key=settings.google_api_key,
    )

    context_str = "\n\n---\n\n".join(
        [
            f"[Firma: {doc.metadata.get('company', '')} | Plik: {doc.metadata.get('filename', '')}]\n{doc.page_content}"
            for doc in documents
        ]
    )

    # 1. Sprawdzanie halucynacji (Groundedness)
    hallucination_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Jesteś audytorem ugruntowania faktograficznego w sprawozdaniach finansowych.\n"
                "Twoim zadaniem jest ocenić, czy wygenerowana odpowiedź wynika z dostarczonych dokumentów źródłowych oraz ich nagłówków/metadanych (nazwa firmy, okres, liczby).\n"
                "Jeśli odpowiedź poprawnie cytuje fakty i liczby ze źródeł (nawet jeśli używa naturalnych parafraz lub odnosi się do nazwy firmy z metadanych), oceń: 'yes'.\n"
                "Tylko jeśli odpowiedź wymyśla liczby lub fakty sprzeczne ze źródłami, oceń: 'no'.",
            ),
            (
                "human",
                "Dokumenty źródłowe wraz z metadanymi:\n{context}\n\nWygenerowana odpowiedź:\n{generation}",
            ),
        ]
    )

    # 2. Sprawdzanie adekwatności odpowiedzi (Answer Relevance)
    answer_grader = llm.with_structured_output(GradeAnswer)
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Jesteś audytorem trafności odpowiedzi. Oceń, czy odpowiedź bezpośrednio i precyzyjnie "
                "rozwiązuje problem zadany w pytaniu użytkownika.\n"
                "Oceń 'yes' jeśli tak, lub 'no' jeśli odpowiedź omija sedno pytania.",
            ),
            (
                "human",
                "Pytanie użytkownika: {question}\n\nUdzielona odpowiedź:\n{generation}",
            ),
        ]
    )

    h_chain = hallucination_prompt | hallucination_grader
    a_chain = answer_prompt | answer_grader

    async def _evaluate_both_concurrently():
        task_h = h_chain.ainvoke({"context": context_str, "generation": generation})
        task_a = a_chain.ainvoke({"question": question, "generation": generation})
        return await asyncio.gather(task_h, task_a, return_exceptions=True)

    # Równoległe wykonanie obu ewaluacji
    results = run_async(_evaluate_both_concurrently())
    h_res, a_res = results[0], results[1]

    hallucination_grade = "grounded"
    if not isinstance(h_res, Exception) and hasattr(h_res, "binary_score"):
        if h_res.binary_score.lower() == "no":
            hallucination_grade = "not grounded"

    answer_grade = "useful"
    if not isinstance(a_res, Exception) and hasattr(a_res, "binary_score"):
        if a_res.binary_score.lower() == "no":
            answer_grade = "not useful"

    return {
        "hallucination_grade": hallucination_grade,
        "answer_grade": answer_grade,
    }
