"""Gradio Web Application dla BigTech Financial Agentic RAG.
Przystosowane do lokalnego uruchomienia oraz hostingu na Hugging Face Spaces.
"""

import os
import sys
from pathlib import Path
from typing import Generator, List, Tuple

import gradio as gr
import torch

# Ścieżka projektu
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_settings
from src.graph import app as agent_app
from src.state import GraphState


def get_system_hardware_badge() -> str:
    """Zwraca informację o aktywnym sprzęcie inferencyjnym."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        return f"⚡ **GPU Aktywne**: `{gpu_name}` ({vram_mb:.0f} MB VRAM) — PyTorch FP16"
    return "💻 **Tryb CPU**: Standardowa inferencja PyTorch FP32 (Hugging Face Spaces Cloud Tier)"


def stream_agentic_rag(
    message: str, history: List[dict]
) -> Generator[Tuple[List[dict], str, str], None, None]:
    """Generator strumieniujący wykonanie cyklicznego grafu LangGraph do interfejsu Gradio."""
    if not message or not message.strip():
        yield history, "*Wpisz zapytanie w polu powyżej.*", ""
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
            "source": "gradio_web_ui",
            "model": settings.gemini_model,
        },
    }

    # Inicjalizacja dymków czatu
    new_history = list(history)
    new_history.append({"role": "user", "content": message.strip()})
    new_history.append(
        {
            "role": "assistant",
            "content": "⏳ *Inicjalizacja grafu stanów LangGraph i przeszukiwanie bazy wektorowej...*",
        }
    )

    status_log = "🚀 **Start Potoku Agentowego**\n"
    sources_preview = "*Pobieranie dokumentów w toku...*"
    yield new_history, status_log, sources_preview

    last_docs = []
    final_generation = ""

    # Strumieniowanie kolejnych węzłów grafu
    for event in agent_app.stream(initial_state, config=config):
        for node_name, state_update in event.items():
            if node_name == "retrieve":
                docs = state_update.get("documents", [])
                last_docs = docs
                status_log += f"🔍 **[retrieve]**: Pobrano {len(docs)} kandydatów (Parent Chunks z tabelami PDF i raportami MD).\n"
                if docs:
                    sources_preview = "\n\n---\n\n".join(
                        [
                            f"**Firma: {d.metadata.get('company', 'Brak')}** | Plik: `{d.metadata.get('filename', 'źródło')}` | Typ: `{d.metadata.get('content_type', 'tekst')}`\n"
                            f"{d.page_content[:400]}..."
                            for d in docs[:3]
                        ]
                    )
                yield new_history, status_log, sources_preview

            elif node_name == "local_rerank":
                scores = state_update.get("rerank_scores", [])
                docs = state_update.get("documents", [])
                last_docs = docs
                top_scores = [round(s, 3) for s in scores[:3]]
                status_log += f"⚡ **[local_rerank]**: Reranker PyTorch przefiltrował fragmenty. Najwyższe oceny: `{top_scores}`.\n"
                yield new_history, status_log, sources_preview

            elif node_name == "grade_documents":
                docs = state_update.get("documents", [])
                last_docs = docs
                needed = state_update.get("web_search_needed", False)
                if needed:
                    status_log += "⚠️ **[grade_documents]**: Odrzucono fragmenty — wymagane przepisanie zapytania.\n"
                else:
                    status_log += f"🤖 **[grade_documents]**: Asynchroniczny LLM Grader (asyncio) zaakceptował `{len(docs)}` relewantnych chunków.\n"
                yield new_history, status_log, sources_preview

            elif node_name == "rewrite_query":
                new_q = state_update.get("question", "")
                retries = state_update.get("retry_count", 0)
                status_log += f"🔄 **[rewrite_query]** *(Autokorekta #{retries})*: Nowe zoptymalizowane zapytanie:\n> *\"{new_q[:100]}...\"*\n"
                yield new_history, status_log, sources_preview

            elif node_name == "generate":
                final_generation = state_update.get("generation", "")
                status_log += "✍️ **[generate]**: Gemini 3 Flash dokonał syntezy odpowiedzi analitycznej.\n"
                new_history[-1]["content"] = final_generation
                yield new_history, status_log, sources_preview

            elif node_name == "hallucination_check":
                h_grade = state_update.get("hallucination_grade", "grounded")
                a_grade = state_update.get("answer_grade", "useful")
                status_log += (
                    f"🛡️ **[hallucination_check]**: Ugruntowanie w faktach: `{h_grade.upper()}` | "
                    f"Celność: `{a_grade.upper()}`\n"
                )
                yield new_history, status_log, sources_preview

    # Zakończenie i dołączenie cytatów w rozwijanym akordeonie
    citations_accordion = ""
    if last_docs:
        citations_accordion = (
            f"\n\n<details><summary><b>📚 Cytowane źródła i tabele finansowe ({len(last_docs)})</b></summary>\n\n"
            f"{sources_preview}\n\n</details>"
        )

    new_history[-1]["content"] = final_generation + citations_accordion
    status_log += "\n✅ **Wykonanie zakończone pomyślnie!** Pełny ślad zarejestrowano w LangSmith."
    yield new_history, status_log, sources_preview


# Budowa interfejsu w Blocks
custom_css = """
footer {visibility: hidden}
.gradio-container {max-width: 1300px !important; margin: auto !important;}
"""

with gr.Blocks(title="BigTech Financial Agentic RAG") as demo:
    gr.Markdown(
        """
        # 🧠 BigTech Financial Agentic RAG — AI Engineering & LLMOps
        ### Samonaprawczy agent analizy sprawozdawczości finansowej (SEC 10-K / 10-Q)
        **Stos technologiczny**: LangGraph • PyTorch CUDA FP16 Cross-Encoder • Gemini 3 Flash • pdfplumber • LangSmith
        """
    )

    with gr.Row():
        hardware_info = gr.Markdown(get_system_hardware_badge())

    with gr.Row():
        # LEWA KOLUMNA: Czat
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                height=520,
                label="Rozmowa z Agentem",
                buttons=["copy"],
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Zadaj precyzyjne pytanie finansowe (np. o marżę NVIDIA, Capex Alphabetu lub chipy AWS)...",
                    label="Pytanie",
                    scale=9,
                    lines=1,
                )
                send_btn = gr.Button("Wyślij 🚀", variant="primary", scale=2)

            with gr.Row():
                clear_btn = gr.Button("Wyczyść historię 🗑️", size="sm")

            gr.Examples(
                examples=[
                    "Porównaj przychody z chmury AWS i Google Cloud w 2024 roku oraz ich rentowność.",
                    "Ile wyniósł zysk netto (Net Income) spółki NVIDIA w Q3 FY2025 według oficjalnego raportu 10-Q?",
                    "Jakie autorskie procesory AI rozwijają Amazon, Google i Microsoft, aby zmniejszyć zależność od GPU NVIDIA?",
                    "Porównaj podejście Apple i Microsoftu do nakładów Capex na infrastrukturę AI w 2024 roku.",
                ],
                inputs=msg_input,
                label="📌 Przykładowe zapytania demonstracyjne",
            )

        # PRAWA KOLUMNA: Inspektor LLMOps & Źródeł
        with gr.Column(scale=5):
            gr.Markdown("### 🔍 Inspektor Działania Agenta (Live Telemetry)")
            status_box = gr.Markdown(
                value="*Zadaj pytanie, aby zobaczyć ślad decyzyjny agenta w czasie rzeczywistym...*",
                label="Kroki Agenta (LangGraph)",
            )

            gr.Markdown("### 📑 Wyodrębnione Fragmenty i Tabele (Parent Chunks)")
            sources_box = gr.Markdown(
                value="*Brak pobranych źródeł.*",
                label="Cytowane Źródła",
            )

            gr.Markdown(
                """
                ---
                **LLMOps & Tracing**:
                - Projekt w LangSmith: [`ai-engineering-lab`](https://smith.langchain.com)
                - Repozytorium: [GitHub: michalkocher-development](https://github.com/michalkocher-development/bigtech-financial-agentic-rag)
                """
            )

    # Rejestracja zdarzeń
    send_event = send_btn.click(
        fn=stream_agentic_rag,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, status_box, sources_box],
    )
    msg_input.submit(
        fn=stream_agentic_rag,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, status_box, sources_box],
    )
    clear_btn.click(lambda: ([], "*Gotowy do pracy.*", "*Brak źródeł.*"), None, [chatbot, status_box, sources_box])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)
