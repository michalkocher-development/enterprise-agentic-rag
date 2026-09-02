"""Węzeł oceny dokumentów (Grade Node) — filtrowanie szumu z uzasadnieniem werdyktu dla każdego chunku."""

import asyncio
from typing import Any, Dict, List, Literal, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import get_settings
from src.state import GraphState


class GradeDocuments(BaseModel):
    """Szczegółowa ocena istotności dokumentu z uzasadnieniem."""

    binary_score: Literal["yes", "no"] = Field(
        description="Czy dokument zawiera informacje istotne dla pytania: 'yes' lub 'no'."
    )
    explanation: str = Field(
        description="Krótkie uzasadnienie (1 zdanie), dlaczego dokument jest relewantny lub dlaczego został odrzucony."
    )


async def _grade_single_doc(evaluator: Any, question: str, doc: Document) -> Tuple[Document, bool, str]:
    """Asynchronicznie ocenia pojedynczy dokument i pobiera uzasadnienie."""
    try:
        score: GradeDocuments = await evaluator.ainvoke(
            {"question": question, "document": doc.page_content}
        )
        is_relevant = score.binary_score.lower() == "yes"
        return doc, is_relevant, score.explanation
    except Exception as e:
        print(f"Błąd podczas asynchronicznej oceny dokumentu: {e}")
        return doc, True, "Zaakceptowano awaryjnie z powodu błędu API ewaluatora."


async def _grade_all_documents(
    evaluator: Any, question: str, documents: List[Document]
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """Równolegle ocenia wszystkie dokumenty i tworzy pełny rejestr werdyktów."""
    tasks = [_grade_single_doc(evaluator, question, doc) for doc in documents]
    results = await asyncio.gather(*tasks)

    survived_docs: List[Document] = []
    verdicts: List[Dict[str, Any]] = []

    for idx, (doc, is_relevant, explanation) in enumerate(results):
        verdict_str = "relevant" if is_relevant else "irrelevant"
        doc.metadata["grader_verdict"] = verdict_str
        doc.metadata["grader_explanation"] = explanation
        chunk_id = doc.metadata.get("chunk_id", f"{doc.metadata.get('filename', 'doc')}#p{idx}")

        verdicts.append(
            {
                "index": idx + 1,
                "chunk_id": chunk_id,
                "filename": doc.metadata.get("filename", "nieznany"),
                "company": doc.metadata.get("company", doc.metadata.get("domain", "ogólny")),
                "preview": doc.page_content[:250],
                "full_text": doc.page_content,
                "is_table": doc.metadata.get("is_table", False),
                "verdict": verdict_str,
                "explanation": explanation,
            }
        )

        if is_relevant:
            survived_docs.append(doc)

    return survived_docs, verdicts


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu weryfikujący jakość i adekwatność dokumentów po rerankingu."""
    settings = get_settings()
    question = state["question"]
    documents = state.get("documents", [])
    lang = state.get("lang", "pl")

    if not documents:
        return {"documents": [], "web_search_needed": True, "graded_verdicts": []}

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
        google_api_key=settings.google_api_key,
    )
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    if lang == "en":
        system_prompt = (
            "You are a rigorous information quality auditor in an Agentic RAG system.\n"
            "Evaluate whether the given text chunk or table contains facts, figures, or context helpful to answering the user question.\n"
            "If the chunk is completely off-topic, strictly choose 'no'.\n"
            "Provide a concise explanation (1 sentence) in English stating why it is useful or why it was rejected."
        )
        human_prompt = "Question: {question}\n\nCandidate chunk:\n{document}\n\nVerdict:"
    else:
        system_prompt = (
            "Jesteś rygorystycznym audytorem jakości informacji w systemie Agentic RAG.\n"
            "Oceniasz, czy dany fragment tekstu lub tabela zawiera fakty, liczby lub kontekst pomocny w odpowiedzi na pytanie użytkownika.\n"
            "Jeśli fragment dotyczy zupełnie innej dziedziny (np. pytanie dotyczy dyplomu/uczelni, a tekst to finanse Microsoftu) — bezwzględnie wybierz 'no'.\n"
            "Podaj zwięzłe uzasadnienie (1 zdanie), dlaczego fragment jest przydatny lub nie."
        )
        human_prompt = "Pytanie: {question}\n\nOceniany fragment:\n{document}\n\nWerdykt:"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    evaluator = prompt | structured_llm_grader

    # Uruchomienie współbieżnej ewaluacji asyncio
    loop = asyncio.new_event_loop()
    try:
        survived_docs, verdicts = loop.run_until_complete(
            _grade_all_documents(evaluator, question, documents)
        )
    finally:
        loop.close()

    web_search_needed = len(survived_docs) == 0

    return {
        "documents": survived_docs,
        "web_search_needed": web_search_needed,
        "graded_verdicts": verdicts,
    }
