"""Węzeł generowania (Generate Node) — precyzyjna synteza odpowiedzi ze zweryfikowanego kontekstu."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.state import GraphState


def generate_node(state: GraphState) -> Dict[str, Any]:
    """Węzeł grafu syntetyzujący odpowiedź na bazie zweryfikowanych dokumentów.

    Generuje rzetelną, opartą na faktach odpowiedź przy użyciu modelu Gemini.
    """
    settings = get_settings()
    question = state["original_question"]
    documents = state.get("documents", [])

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
        google_api_key=settings.google_api_key,
    )

    context_str = "\n\n---\n\n".join(
        [f"[Dokument: {doc.metadata.get('filename', 'źródło')}]\n{doc.page_content}" for doc in documents]
    )

    system_prompt = (
        "Jesteś precyzyjnym analitykiem finansowym rynków kapitałowych i technologii AI.\n"
        "Odpowiadasz na pytania użytkownika WYŁĄCZNIE na podstawie dostarczonego poniżej kontekstu ze sprawozdań finansowych.\n"
        "Zasady bezwzględne:\n"
        "1. Podawaj dokładne liczby, waluty, kwartały i lata podatkowe dokładnie tak, jak podano w źródłach.\n"
        "2. Jeśli kontekst nie zawiera odpowiedzi na jakieś pytanie, wprost napisz, że dane źródłowe nie zawierają tej informacji.\n"
        "3. Nie twórz przypuszczeń ani nie uzupełniaj faktów spoza dostarczonego tekstu.\n"
        "4. Odpowiadaj w języku polskim, zwięźle i profesjonalnie."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Kontekst ze sprawozdań finansowych:\n{context}\n\nPytanie: {question}\n\nOdpowiedź analityczna:",
            ),
        ]
    )

    reg_count = state.get("regeneration_count", 0)
    if state.get("generation"):
        reg_count += 1

    chain = prompt | llm | StrOutputParser()
    generation = str(chain.invoke({"context": context_str, "question": question})).strip()

    return {
        "generation": generation,
        "regeneration_count": reg_count,
    }
