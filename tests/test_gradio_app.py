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
    assert "GPU Aktywne" in badge or "Tryb CPU" in badge


def test_stream_agentic_generator_empty_input():
    """Weryfikuje zachowanie generatora przy pustym wejściu."""
    history = []
    generator = stream_agentic_rag("", history)
    result = next(generator)
    
    assert len(result) == 3
    updated_history, status, sources = result
    assert "Wpisz zapytanie" in status
