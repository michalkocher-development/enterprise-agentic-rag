"""Testy weryfikujące poprawność aplikacji webowej Gradio (app.py)."""

import pytest
import gradio as gr
from app import demo, stream_agentic_rag, get_system_hardware_badge


def test_gradio_blocks_initialization():
    """Weryfikuje, czy aplikacja Gradio tworzy poprawny obiekt gr.Blocks z komponentami."""
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "BigTech Financial Agentic RAG"


def test_hardware_badge_detection():
    """Weryfikuje detekcję sprzętu (GPU RTX 2050 lub fallback na CPU)."""
    badge = get_system_hardware_badge()
    assert "GPU:" in badge or "CPU:" in badge


def test_stream_agentic_generator_empty_input():
    """Weryfikuje zachowanie generatora przy pustym wejściu."""
    history = []
    generator = stream_agentic_rag("", history)
    result = next(generator)
    assert result == history


def test_stream_agentic_generator_starts_reasoning():
    """Weryfikuje, czy generator tworzy krok myślenia agenta w akordeonie."""
    history = []
    generator = stream_agentic_rag("Test query", history)
    result = next(generator)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert "<details" in result[1]["content"]

