"""Gradio Web Application dla BigTech Financial Agentic RAG.
Układ centralny (Perplexity / Claude style) z wyśrodkowaną kolumną i wbudowanym śladem myślenia.
"""

import os
import sys
from pathlib import Path
from typing import Generator, List

import gradio as gr
import torch

# Ścieżka projektu
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_settings
from src.graph import app as agent_app
from src.state import GraphState


def get_hardware_pills() -> str:
    """Zwraca pigułki statusowe dla górnej belki."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        gpu_pill = f"`🟢 GPU: {gpu_name} ({vram_mb:.0f} MB VRAM — FP16)`"
    else:
        gpu_pill = "`💻 CPU: PyTorch FP32 Fallback`"

    return f"{gpu_pill} &nbsp;•&nbsp; `📊 Baza: SEC 10-K/10-Q Big 6` &nbsp;•&nbsp; `🤖 LLM: Gemini 3 Flash` &nbsp;•&nbsp; `🔗 LangSmith Active`"


get_system_hardware_badge = get_hardware_pills


def format_steps_details(steps: List[str], is_running: bool = True) -> str:
    """Formatuje listę kroków myślenia agenta w zwijany akordeon dopasowany do trybu Dark i Light."""
    if not steps:
        return ""
    
    icon = "⏳" if is_running else "✅"
    title = f"{icon} Ślad myślenia agenta ({len(steps)} kroków grafu LangGraph)"
    state_attr = "open" if is_running else ""
    
    items = "\n".join([f"- {step}" for step in steps])
    return (
        f"<details {state_attr} class='agent-steps-box'>\n"
        f"<summary>{title}</summary>\n\n"
        f"{items}\n</details>\n\n"
    )



def stream_agentic_rag(
    message: str, history: List[dict]
) -> Generator[List[dict], None, None]:
    """Generator strumieniujący działanie grafu LangGraph bezpośrednio do wyśrodkowanego czatu."""
    if not message or not message.strip():
        yield history
        return

    settings = get_settings()

    initial_state: GraphState = {
        "question": message.strip(),
        "original_question": message.strip(),
        "documents": [],
        "rerank_scores": [],
        "generation": None,
        "retry_count": 0,
        "regeneration_count": 0,
        "hallucination_grade": None,
        "answer_grade": None,
        "web_search_needed": False,
    }

    config = {
        "configurable": {
            "thread_id": f"gradio-session-{os.urandom(4).hex()}",
        },
        "metadata": {
            "source": "gradio_centered_ui",
            "model": settings.gemini_model,
        },
    }

    # Przygotowanie historii
    new_history = list(history)
    new_history.append({"role": "user", "content": message.strip()})
    
    steps: List[str] = ["🚀 *Inicjalizacja pętli agentowej LangGraph...*"]
    new_history.append(
        {
            "role": "assistant",
            "content": format_steps_details(steps, is_running=True) + "*Przeszukiwanie bazy wektorowej...*",
        }
    )
    yield new_history

    last_docs = []
    final_generation = ""

    for event in agent_app.stream(initial_state, config=config):
        for node_name, state_update in event.items():
            if node_name == "retrieve":
                docs = state_update.get("documents", [])
                last_docs = docs
                steps.append(f"🔍 **[retrieve]**: Pobrano `{len(docs)}` kandydatów (Parent Chunks z tabelami PDF i raportami MD).")
                new_history[-1]["content"] = (
                    format_steps_details(steps, is_running=True) + "*Przeliczanie wag przez Cross-Encoder na GPU...*"
                )
                yield new_history

            elif node_name == "local_rerank":
                scores = state_update.get("rerank_scores", [])
                docs = state_update.get("documents", [])
                last_docs = docs
                top_scores = [round(s, 3) for s in scores[:3]]
                steps.append(f"⚡ **[local_rerank]**: Reranker PyTorch przefiltrował fragmenty (Top oceny: `{top_scores}`).")
                new_history[-1]["content"] = (
                    format_steps_details(steps, is_running=True) + "*Asynchroniczna weryfikacja relewantności (asyncio)...*"
                )
                yield new_history

            elif node_name == "grade_documents":
                docs = state_update.get("documents", [])
                last_docs = docs
                needed = state_update.get("web_search_needed", False)
                if needed:
                    steps.append("⚠️ **[grade_documents]**: Odrzucono fragmenty — zapytanie wymaga autokorekty.")
                else:
                    steps.append(f"🤖 **[grade_documents]**: LLM Grader zaakceptował `{len(docs)}` relewantnych chunków.")
                new_history[-1]["content"] = (
                    format_steps_details(steps, is_running=True) + "*Generowanie odpowiedzi przez Gemini 3 Flash...*"
                )
                yield new_history

            elif node_name == "rewrite_query":
                new_q = state_update.get("question", "")
                retries = state_update.get("retry_count", 0)
                steps.append(f"🔄 **[rewrite_query]** *(Korekta #{retries})*: Zoptymalizowano zapytanie:\n  > *\"{new_q[:90]}...\"*")
                new_history[-1]["content"] = (
                    format_steps_details(steps, is_running=True) + "*Ponowne wyszukiwanie z poprawionym zapytaniem...*"
                )
                yield new_history

            elif node_name == "generate":
                final_generation = state_update.get("generation", "")
                steps.append("✍️ **[generate]**: Gemini 3 Flash dokonał syntezy odpowiedzi analitycznej.")
                new_history[-1]["content"] = (
                    format_steps_details(steps, is_running=True) + final_generation
                )
                yield new_history

            elif node_name == "hallucination_check":
                h_grade = state_update.get("hallucination_grade", "grounded")
                a_grade = state_update.get("answer_grade", "useful")
                steps.append(f"🛡️ **[hallucination_check]**: Ugruntowanie w faktach: `{h_grade.upper()}` | Celność: `{a_grade.upper()}`")
                new_history[-1]["content"] = (
                    format_steps_details(steps, is_running=True) + final_generation
                )
                yield new_history

    # Zakończenie: zwijamy akordeon myślenia i dodajemy cytaty
    sources_accordion = ""
    if last_docs:
        formatted_sources = "\n\n---\n\n".join(
            [
                f"**Firma: {d.metadata.get('company', 'Brak')}** | Plik: `{d.metadata.get('filename', 'źródło')}` | Typ: `{d.metadata.get('content_type', 'tekst')}`\n\n"
                f"{d.page_content}"
                for d in last_docs[:4]
            ]
        )
        sources_accordion = (
            f"\n\n<details class='sources-box'>"
            f"<summary>📚 Cytowane źródła i tabele finansowe ({len(last_docs)})</summary>\n\n"
            f"{formatted_sources}\n\n</details>"
        )

    # Zamykamy akordeon myślenia po zakończeniu (is_running=False)
    new_history[-1]["content"] = (
        format_steps_details(steps, is_running=False)
        + final_generation
        + sources_accordion
    )
    yield new_history


# CSS wyśrodkowujący i nadający lekki, przestronny charakter
custom_css = """
.gradio-container {
    max-width: 940px !important;
    margin: 0 auto !important;
    padding: 20px 15px !important;
}
.header-box {
    text-align: center;
    margin-bottom: 20px;
}
.pills-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 8px;
    margin-bottom: 12px;
}
.quick-chip {
    text-align: left !important;
    font-size: 0.85em !important;
    padding: 8px 12px !important;
}
.agent-steps-box {
    margin-bottom: 14px;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid rgba(128, 128, 128, 0.28);
    background-color: rgba(128, 128, 128, 0.12);
    font-size: 0.9em;
    color: inherit;
}
.agent-steps-box summary {
    cursor: pointer;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 8px;
}
.agent-steps-box ul, .agent-steps-box li {
    color: inherit !important;
    margin-top: 4px;
}
.sources-box {
    margin-top: 16px;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid rgba(128, 128, 128, 0.28);
    background-color: rgba(128, 128, 128, 0.12);
    font-size: 0.9em;
    color: inherit;
}
.sources-box summary {
    cursor: pointer;
    font-weight: 600;
    color: #34d399;
    margin-bottom: 8px;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(title="BigTech Financial Agentic RAG") as demo:
    # 1. Górna Belka (Header & Status Pills)
    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            """
            # 🧠 BigTech Financial Agentic RAG
            ### Inteligentna analiza sprawozdawczości finansowej SEC 10-K / 10-Q (Big 6)
            """
        )
        gr.Markdown(get_hardware_pills())

    # 2. Główny Wyśrodkowany Czat
    chatbot = gr.Chatbot(
        height=540,
        label="Rozmowa z Analitykiem AI",
        buttons=["copy"],
    )

    # 3. Pole Wprowadzania Zapytania
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Zadaj pytanie finansowe (np. o zysk netto NVIDIA z tabeli 10-Q, Capex Alphabetu lub chipy AWS)...",
            show_label=False,
            scale=9,
            lines=1,
            autofocus=True,
        )
        send_btn = gr.Button("Wyślij 🚀", variant="primary", scale=2)
        clear_btn = gr.Button("Wyczyść 🗑️", size="sm", scale=1)

    # 4. Szybkie Chipy Pytaniowe (2x2 Grid)
    gr.Markdown("##### 💡 Szybkie zapytania demonstracyjne (kliknij, aby zapytać):")
    with gr.Row():
        chip1 = gr.Button("📊 Zysk netto (Net Income) NVIDIA w Q3 FY25 z tabeli 10-Q", size="sm", elem_classes=["quick-chip"])
        chip2 = gr.Button("☁️ Porównaj przychody AWS i Google Cloud w 2024 roku oraz ich rentowność", size="sm", elem_classes=["quick-chip"])
    with gr.Row():
        chip3 = gr.Button("⚡ Jakie autorskie procesory AI rozwijają Amazon, Google i Microsoft?", size="sm", elem_classes=["quick-chip"])
        chip4 = gr.Button("💰 Porównaj nakłady Capex na centra danych Apple i Microsoftu w 2024", size="sm", elem_classes=["quick-chip"])

    # Obsługa kliknięć chipów
    def run_chip_query(query_text: str, history: List[dict]):
        yield from stream_agentic_rag(query_text, history)

    chip1.click(lambda: "Ile wyniósł zysk netto (Net Income) spółki NVIDIA w Q3 FY2025 według oficjalnego raportu 10-Q?", None, msg_input).then(
        fn=stream_agentic_rag, inputs=[msg_input, chatbot], outputs=[chatbot]
    )
    chip2.click(lambda: "Porównaj przychody z chmury AWS i Google Cloud w 2024 roku oraz ich rentowność.", None, msg_input).then(
        fn=stream_agentic_rag, inputs=[msg_input, chatbot], outputs=[chatbot]
    )
    chip3.click(lambda: "Jakie autorskie procesory AI rozwijają Amazon, Google i Microsoft, aby zmniejszyć zależność od GPU NVIDIA?", None, msg_input).then(
        fn=stream_agentic_rag, inputs=[msg_input, chatbot], outputs=[chatbot]
    )
    chip4.click(lambda: "Porównaj podejście Apple i Microsoftu do nakładów Capex na infrastrukturę AI w 2024 roku.", None, msg_input).then(
        fn=stream_agentic_rag, inputs=[msg_input, chatbot], outputs=[chatbot]
    )

    # Rejestracja zdarzeń wprowadzania
    send_btn.click(
        fn=stream_agentic_rag,
        inputs=[msg_input, chatbot],
        outputs=[chatbot],
    ).then(lambda: "", None, msg_input)

    msg_input.submit(
        fn=stream_agentic_rag,
        inputs=[msg_input, chatbot],
        outputs=[chatbot],
    ).then(lambda: "", None, msg_input)

    clear_btn.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=custom_css,
    )
