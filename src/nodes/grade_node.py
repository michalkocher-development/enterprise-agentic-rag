"""Węzeł oceny dokumentów (Grade Node) — filtrowanie szumu przez równoległy LLM Grader (asyncio)."""

import asyncio
from typing import Any, Dict, List, Literal, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import get_settings
from src.state import GraphState
from src.utils.async_runner import run_async


class GradeDocuments(BaseModel):
    """Binarna ocena istotności odnalezionego dokumentu względem pytania użytkownika."""

    binary_score: Literal["yes", "no"] = Field(
        description="Czy dokument zawiera informacje istotne dla pytania: 'yes' lub 'no'."
    )


async def _grade_single_doc(evaluator: Any, question: str, doc: Document) -> Tuple[Document, bool]:
    """Asynchronicznie ocenia pojedynczy dokument."""
    try:
        score: GradeDocuments = await evaluator.ainvoke(
            {"question": question, "document": doc.page_content}
        )
        return doc, (score.binary_score.lower() == "yes")
    except Exception as e:
        print(f"Błąd podczas asynchronicznej oceny dokumentu: {e}")
        return doc, True  # W razie błędu API zachowujemy dokument bezpiecznie


async def _grade_all_documents(evaluator: Any, question: str, documents: List[Document]) -> List[Document]:
    """Równolegle ocenia wszystkie dokumenty za pomocą asyncio.gather."""
    tasks = [_grade_single_doc(evaluator, question, doc) for doc in documents]
    results = await asyncio.gather(*tasks)
    return [doc for doc, is_relevant in results if is_relevant]


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu weryfikujący jakość i adekwatność dokumentów po rerankingu.
    
    Wykorzystuje pełną współbieżność asyncio do jednoczesnej ewaluacji wszystkich chunków w sieci.
    """
    settings = get_settings()
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {"documents": [], "web_search_needed": True}

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
        google_api_key=settings.google_api_key,
    )
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system_prompt = (
        "Jesteś precyzyjnym audytorem merytorycznym sprawozdawczości finansowej Big Tech.\n"
        "Twoim zadaniem jest ocena, czy podany fragment dokumentu zawiera fakty lub dane liczbowe "
        "pomocne w odpowiedzi na pytanie użytkownika.\n"
        "Oceń binarnie: 'yes' jeśli fragment jest relewantny, 'no' jeśli jest nieprzydatny lub niezwiązany z pytaniem."
    )

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Pytanie użytkownika: {question}\n\nOceniany fragment dokumentu:\n{document}",
            ),
        ]
    )

    evaluator = grade_prompt | structured_llm_grader

    # Równoległa ewaluacja chunków za pomocą asyncio.gather
    filtered_docs = run_async(_grade_all_documents(evaluator, question, documents))
    web_search_needed = len(filtered_docs) == 0

    return {
        "documents": filtered_docs,
        "web_search_needed": web_search_needed,
    }
