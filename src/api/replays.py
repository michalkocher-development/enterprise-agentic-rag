"""Moduł odtwarzania zapisanych wzorcowych przebiegów SSE (Replay Engine).

Zapewnia 100% determinizm i zerowy koszt API podczas prezentacji portfolio rekruterom
lub testów wizualnych UI bez konieczności ciągłego odpalania GPU i Gemini API.
"""

import asyncio
import json
from typing import AsyncGenerator, Dict

REPLAY_SCENARIOS: Dict[str, Dict] = {
    "direct_answer": {
        "id": "direct_answer",
        "title": "Konwersacja z pamięci (Direct Answer)",
        "description": "Demonstruje Adaptive Routing: router wykrywa brak potrzeby RAG i generuje odpowiedź z pamięci dialogu.",
        "query": "W czym możesz mi pomóc i jakie masz możliwości?",
        "has_self_correction": False,
        "estimated_duration_s": 2.2,
        "events": [
            ("session", {"thread_id": "replay-direct-answer"}),
            ("node_start", {"node": "router", "edge_from": "START", "iteration": 0}),
            (
                "step",
                {
                    "node": "router",
                    "latency_ms": 320.0,
                    "total_ms": 320.0,
                    "route": "direct_answer",
                    "reasoning": "Zapytanie ogólne o charakterze konwersacyjnym — odpowiedź zostanie udzielona bezpośrednio z pamięci asystenta bez angażowania bazy dokumentacji.",
                },
            ),
            ("node_start", {"node": "generate", "edge_from": "router", "iteration": 0}),
            ("token", {"node": "generate", "delta": "Cześć! "}),
            ("token", {"node": "generate", "delta": "Jestem autonomicznym asystentem inżynieryjnym "}),
            ("token", {"node": "generate", "delta": "platformy **Enterprise Agentic RAG**.\n\n"}),
            ("token", {"node": "generate", "delta": "Posiadam następujące możliwości:\n"}),
            ("token", {"node": "generate", "delta": "1. **Adaptive Routing**: Rozpoznaję pytania wymagające głębokiego przeszukania bazy wiedzy vs rozmowę bieżącą.\n"}),
            ("token", {"node": "generate", "delta": "2. **Lokalny Reranking GPU**: Akceleruję scoring kandydatów modelem Cross-Encoder na NVIDIA GeForce RTX 2050 (FP16).\n"}),
            ("token", {"node": "generate", "delta": "3. **Self-Correction**: Potrafię automatycznie przeformułować zapytanie, jeśli pierwsze wyniki nie spełniają kryteriów jakości.\n"}),
            ("token", {"node": "generate", "delta": "4. **Audyt Halucynacji**: Każda synteza przechodzi podwójną weryfikację ugruntowania faktograficznego."}),
            (
                "step",
                {
                    "node": "generate",
                    "latency_ms": 1150.0,
                    "total_ms": 1470.0,
                    "generation": (
                        "Cześć! Jestem autonomicznym asystentem inżynieryjnym platformy **Enterprise Agentic RAG**.\n\n"
                        "Posiadam następujące możliwości:\n"
                        "1. **Adaptive Routing**: Rozpoznaję pytania wymagające głębokiego przeszukania bazy wiedzy vs rozmowę bieżącą.\n"
                        "2. **Lokalny Reranking GPU**: Akceleruję scoring kandydatów modelem Cross-Encoder na NVIDIA GeForce RTX 2050 (FP16).\n"
                        "3. **Self-Correction**: Potrafię automatycznie przeformułować zapytanie, jeśli pierwsze wyniki nie spełniają kryteriów jakości.\n"
                        "4. **Audyt Halucynacji**: Każda synteza przechodzi podwójną weryfikację ugruntowania faktograficznego."
                    ),
                    "citations": [],
                },
            ),
            ("complete", {"status": "done", "thread_id": "replay-direct-answer", "total_time_ms": 1470.0}),
        ],
    },
    "standard_rag": {
        "id": "standard_rag",
        "title": "Precyzyjny RAG Big Tech (NVIDIA 10-Q)",
        "description": "Pełny przepływ Parent-Document Retrieval, scoring GPU RTX 2050, LLM Grader i ekstrakcja tabeli finansowej.",
        "query": "Jaki był zysk netto i marża brutto NVIDIA w Q3 FY2025 według oficjalnego raportu 10-Q?",
        "has_self_correction": False,
        "estimated_duration_s": 5.8,
        "events": [
            ("session", {"thread_id": "replay-standard-rag"}),
            ("node_start", {"node": "router", "edge_from": "START", "iteration": 0}),
            ("step", {"node": "router", "latency_ms": 310.0, "total_ms": 310.0, "route": "retrieve"}),
            ("node_start", {"node": "retrieve", "edge_from": "router", "iteration": 0}),
            (
                "step",
                {
                    "node": "retrieve",
                    "latency_ms": 420.0,
                    "total_ms": 730.0,
                    "documents_count": 10,
                    "candidates": [
                        {"index": 1, "chunk_id": "nvidia_q3_fy25_10q#t1", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "is_table": True, "preview": "| Metric | Q3 FY25 | Q3 FY24 |\n| Net Income | $19,309M | $9,243M |\n| Gross Margin | 74.6% | 74.0% |"},
                        {"index": 2, "chunk_id": "nvidia_q3_fy25_10q#p2", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "is_table": False, "preview": "Net income for the third quarter of fiscal 2025 was $19,309 million, compared with $9,243 million a year ago."},
                        {"index": 3, "chunk_id": "nvidia_fy25#p0", "filename": "nvidia_fy25.md", "company": "NVIDIA", "is_table": False, "preview": "NVIDIA GAAP gross margin for the quarter was 74.6% and non-GAAP gross margin was 75.1%."},
                        {"index": 4, "chunk_id": "alphabet_fy24_fy25#p1", "filename": "alphabet_fy24_fy25.md", "company": "Alphabet", "is_table": False, "preview": "Alphabet revenues grew 15% year-over-year with operating margin reaching 32%."},
                        {"index": 5, "chunk_id": "microsoft_fy24_fy25#p3", "filename": "microsoft_fy24_fy25.md", "company": "Microsoft", "is_table": False, "preview": "Microsoft Cloud revenue was $38.9 billion, up 22% year-over-year."},
                        {"index": 6, "chunk_id": "apple_fy24_fy25#p2", "filename": "apple_fy24_fy25.md", "company": "Apple", "is_table": False, "preview": "Apple quarterly revenue of $94.9 billion, up 6 percent year over year."},
                        {"index": 7, "chunk_id": "amazon_fy24_fy25#p0", "filename": "amazon_fy24_fy25.md", "company": "Amazon", "is_table": False, "preview": "AWS segment sales increased 19% year-over-year to $27.5 billion."},
                        {"index": 8, "chunk_id": "meta_fy24_fy25#p1", "filename": "meta_fy24_fy25.md", "company": "Meta", "is_table": False, "preview": "Meta capital expenditures for the full year 2024 are expected to be $38-40 billion."},
                        {"index": 9, "chunk_id": "nvidia_q3_fy25_10q#p5", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "is_table": False, "preview": "Cash, cash equivalents and marketable securities were $38.5 billion."},
                        {"index": 10, "chunk_id": "eu_ai_act_art5#p0", "filename": "eu_ai_act_art5.md", "company": "EU Governance", "is_table": False, "preview": "Article 5 of the EU AI Act defines prohibited artificial intelligence practices."},
                    ],
                },
            ),
            ("node_start", {"node": "local_rerank", "edge_from": "retrieve", "iteration": 0}),
            (
                "step",
                {
                    "node": "local_rerank",
                    "latency_ms": 780.0,
                    "total_ms": 1510.0,
                    "gpu_device": "NVIDIA GeForce RTX 2050 (FP16)",
                    "vram_mb": 428.5,
                    "ranked_candidates": [
                        {"rank": 1, "chunk_id": "nvidia_q3_fy25_10q#t1", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "score": 0.984, "kept": True, "is_table": True, "preview": "| Metric | Q3 FY25 | Q3 FY24 |\n| Net Income | $19,309M | $9,243M |\n| Gross Margin | 74.6% | 74.0% |"},
                        {"rank": 2, "chunk_id": "nvidia_q3_fy25_10q#p2", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "score": 0.952, "kept": True, "is_table": False, "preview": "Net income for the third quarter of fiscal 2025 was $19,309 million..."},
                        {"rank": 3, "chunk_id": "nvidia_fy25#p0", "filename": "nvidia_fy25.md", "company": "NVIDIA", "score": 0.915, "kept": True, "is_table": False, "preview": "NVIDIA GAAP gross margin for the quarter was 74.6%..."},
                        {"rank": 4, "chunk_id": "nvidia_q3_fy25_10q#p5", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "score": 0.612, "kept": True, "is_table": False, "preview": "Cash, cash equivalents and marketable securities were $38.5 billion."},
                        {"rank": 5, "chunk_id": "alphabet_fy24_fy25#p1", "filename": "alphabet_fy24_fy25.md", "company": "Alphabet", "score": 0.142, "kept": False, "is_table": False, "preview": "Alphabet revenues grew 15%..."},
                        {"rank": 6, "chunk_id": "microsoft_fy24_fy25#p3", "filename": "microsoft_fy24_fy25.md", "company": "Microsoft", "score": 0.118, "kept": False, "is_table": False, "preview": "Microsoft Cloud revenue was $38.9 billion..."},
                        {"rank": 7, "chunk_id": "apple_fy24_fy25#p2", "filename": "apple_fy24_fy25.md", "company": "Apple", "score": 0.095, "kept": False, "is_table": False, "preview": "Apple quarterly revenue of $94.9 billion..."},
                        {"rank": 8, "chunk_id": "amazon_fy24_fy25#p0", "filename": "amazon_fy24_fy25.md", "company": "Amazon", "score": 0.081, "kept": False, "is_table": False, "preview": "AWS segment sales increased 19%..."},
                        {"rank": 9, "chunk_id": "meta_fy24_fy25#p1", "filename": "meta_fy24_fy25.md", "company": "Meta", "score": 0.065, "kept": False, "is_table": False, "preview": "Meta capital expenditures..."},
                        {"rank": 10, "chunk_id": "eu_ai_act_art5#p0", "filename": "eu_ai_act_art5.md", "company": "EU Governance", "score": 0.012, "kept": False, "is_table": False, "preview": "Article 5 of the EU AI Act defines..."},
                    ],
                },
            ),
            ("node_start", {"node": "grade_documents", "edge_from": "local_rerank", "iteration": 0}),
            (
                "step",
                {
                    "node": "grade_documents",
                    "latency_ms": 650.0,
                    "total_ms": 2160.0,
                    "accepted_count": 3,
                    "rejected_count": 1,
                    "graded_verdicts": [
                        {"index": 1, "chunk_id": "nvidia_q3_fy25_10q#t1", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "verdict": "relevant", "explanation": "Zawiera dokładne liczby zysku netto ($19,309M) oraz marży brutto (74.6%) w tabeli Q3 FY25."},
                        {"index": 2, "chunk_id": "nvidia_q3_fy25_10q#p2", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "verdict": "relevant", "explanation": "Potwierdza w tekście narracyjnym wartość zysku netto za Q3 FY25."},
                        {"index": 3, "chunk_id": "nvidia_fy25#p0", "filename": "nvidia_fy25.md", "company": "NVIDIA", "verdict": "relevant", "explanation": "Precyzuje marżę brutto GAAP (74.6%) i non-GAAP (75.1%)."},
                        {"index": 4, "chunk_id": "nvidia_q3_fy25_10q#p5", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "verdict": "irrelevant", "explanation": "Dotyczy stanu środków pieniężnych i papierów wartościowych, brak danych o marży i zysku."},
                    ],
                },
            ),
            ("node_start", {"node": "generate", "edge_from": "grade_documents", "iteration": 0}),
            ("token", {"node": "generate", "delta": "Według oficjalnego sprawozdania finansowego **NVIDIA Form 10-Q** za trzeci kwartał roku obrotowego 2025 (Q3 FY2025 zakończony 27 października 2024 r.):\n\n"}),
            ("token", {"node": "generate", "delta": "### Kluczowe Wskaźniki Finansowe:\n"}),
            ("token", {"node": "generate", "delta": "| Wskaźnik | Q3 FY2025 | Q3 FY2024 | Zmiana r/r |\n"}),
            ("token", {"node": "generate", "delta": "| :--- | :--- | :--- | :--- |\n"}),
            ("token", {"node": "generate", "delta": "| **Zysk netto (GAAP Net Income)** | **$19,309 mln** ($19.31 mld) | $9,243 mln | **+109%** |\n"}),
            ("token", {"node": "generate", "delta": "| **Marża brutto (GAAP Gross Margin)** | **74.6%** | 74.0% | **+60 pb** |\n"}),
            ("token", {"node": "generate", "delta": "| **Marża brutto non-GAAP** | **75.1%** | 75.0% | **+10 pb** |\n\n"}),
            ("token", {"node": "generate", "delta": "Wzrost zysku netto o ponad 100% był napędzany bezprecedensowym popytem na układy architektur Hopper (H100/H200) oraz wczesnymi dostawami Blackwell w segmencie Data Center."}),
            (
                "step",
                {
                    "node": "generate",
                    "latency_ms": 1820.0,
                    "total_ms": 3980.0,
                    "generation": (
                        "Według oficjalnego sprawozdania finansowego **NVIDIA Form 10-Q** za trzeci kwartał roku obrotowego 2025 (Q3 FY2025 zakończony 27 października 2024 r.):\n\n"
                        "### Kluczowe Wskaźniki Finansowe:\n"
                        "| Wskaźnik | Q3 FY2025 | Q3 FY2024 | Zmiana r/r |\n"
                        "| :--- | :--- | :--- | :--- |\n"
                        "| **Zysk netto (GAAP Net Income)** | **$19,309 mln** ($19.31 mld) | $9,243 mln | **+109%** |\n"
                        "| **Marża brutto (GAAP Gross Margin)** | **74.6%** | 74.0% | **+60 pb** |\n"
                        "| **Marża brutto non-GAAP** | **75.1%** | 75.0% | **+10 pb** |\n\n"
                        "Wzrost zysku netto o ponad 100% był napędzany bezprecedensowym popytem na układy architektur Hopper (H100/H200) oraz wczesnymi dostawami Blackwell w segmencie Data Center."
                    ),
                    "citations": [
                        {"chunk_id": "nvidia_q3_fy25_10q#t1", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "is_table": True, "snippet": "| Metric | Q3 FY25 | Q3 FY24 |\n| Net Income | $19,309M | $9,243M |\n| Gross Margin | 74.6% | 74.0% |"},
                        {"chunk_id": "nvidia_q3_fy25_10q#p2", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "is_table": False, "snippet": "Net income for the third quarter of fiscal 2025 was $19,309 million, compared with $9,243 million a year ago."},
                        {"chunk_id": "nvidia_fy25#p0", "filename": "nvidia_fy25.md", "company": "NVIDIA", "is_table": False, "snippet": "NVIDIA GAAP gross margin for the quarter was 74.6% and non-GAAP gross margin was 75.1%."},
                    ],
                },
            ),
            ("node_start", {"node": "hallucination_check", "edge_from": "generate", "iteration": 0}),
            (
                "step",
                {
                    "node": "hallucination_check",
                    "latency_ms": 720.0,
                    "total_ms": 4700.0,
                    "hallucination_grade": "grounded",
                    "answer_grade": "useful",
                },
            ),
            ("complete", {"status": "done", "thread_id": "replay-standard-rag", "total_time_ms": 4700.0}),
        ],
    },
    "self_correction": {
        "id": "self_correction",
        "title": "Pętla Samonaprawcza (Self-Correction Loop)",
        "description": "Demonstruje zachowanie agenta przy potocznym zapytaniu: odrzucenie szumu przez LLM Grader, przepisanie zapytania i odnalezienie faktów w 2. iteracji.",
        "query": "Ile NVIDIA zarobiła na kartach do grania w Q3 FY2025?",
        "has_self_correction": True,
        "estimated_duration_s": 8.5,
        "events": [
            ("session", {"thread_id": "replay-self-correction"}),
            ("node_start", {"node": "router", "edge_from": "START", "iteration": 0}),
            ("step", {"node": "router", "latency_ms": 290.0, "total_ms": 290.0, "route": "retrieve"}),
            ("node_start", {"node": "retrieve", "edge_from": "router", "iteration": 0}),
            (
                "step",
                {
                    "node": "retrieve",
                    "latency_ms": 410.0,
                    "total_ms": 700.0,
                    "documents_count": 10,
                    "candidates": [
                        {"index": 1, "chunk_id": "nvidia_fy25#p3", "filename": "nvidia_fy25.md", "company": "NVIDIA", "is_table": False, "preview": "NVIDIA Professional Visualization segment revenue was $486 million..."},
                        {"index": 2, "chunk_id": "apple_fy24_fy25#p1", "filename": "apple_fy24_fy25.md", "company": "Apple", "is_table": False, "preview": "Apple Arcade and gaming ecosystem services grew double digits..."},
                        {"index": 3, "chunk_id": "microsoft_fy24_fy25#p4", "filename": "microsoft_fy24_fy25.md", "company": "Microsoft", "is_table": False, "preview": "Xbox content and services revenue increased 61% driven by Activision..."},
                        {"index": 4, "chunk_id": "amazon_fy24_fy25#p2", "filename": "amazon_fy24_fy25.md", "company": "Amazon", "is_table": False, "preview": "Amazon Prime Gaming and Luna cloud gaming integration..."},
                    ],
                },
            ),
            ("node_start", {"node": "local_rerank", "edge_from": "retrieve", "iteration": 0}),
            (
                "step",
                {
                    "node": "local_rerank",
                    "latency_ms": 740.0,
                    "total_ms": 1440.0,
                    "gpu_device": "NVIDIA GeForce RTX 2050 (FP16)",
                    "vram_mb": 425.0,
                    "ranked_candidates": [
                        {"rank": 1, "chunk_id": "nvidia_fy25#p3", "filename": "nvidia_fy25.md", "company": "NVIDIA", "score": 0.482, "kept": True, "is_table": False, "preview": "NVIDIA Professional Visualization..."},
                        {"rank": 2, "chunk_id": "apple_fy24_fy25#p1", "filename": "apple_fy24_fy25.md", "company": "Apple", "score": 0.410, "kept": True, "is_table": False, "preview": "Apple Arcade..."},
                        {"rank": 3, "chunk_id": "microsoft_fy24_fy25#p4", "filename": "microsoft_fy24_fy25.md", "company": "Microsoft", "score": 0.395, "kept": True, "is_table": False, "preview": "Xbox content..."},
                        {"rank": 4, "chunk_id": "amazon_fy24_fy25#p2", "filename": "amazon_fy24_fy25.md", "company": "Amazon", "score": 0.312, "kept": True, "is_table": False, "preview": "Amazon Prime Gaming..."},
                    ],
                },
            ),
            ("node_start", {"node": "grade_documents", "edge_from": "local_rerank", "iteration": 0}),
            (
                "step",
                {
                    "node": "grade_documents",
                    "latency_ms": 690.0,
                    "total_ms": 2130.0,
                    "accepted_count": 0,
                    "rejected_count": 4,
                    "web_search_needed": True,
                    "graded_verdicts": [
                        {"index": 1, "chunk_id": "nvidia_fy25#p3", "filename": "nvidia_fy25.md", "company": "NVIDIA", "verdict": "irrelevant", "explanation": "Odrzucono: fragment dotyczy Professional Visualization, a nie kart do gier."},
                        {"index": 2, "chunk_id": "apple_fy24_fy25#p1", "filename": "apple_fy24_fy25.md", "company": "Apple", "verdict": "irrelevant", "explanation": "Odrzucono: dotyczy usług Apple Arcade, inna spółka."},
                        {"index": 3, "chunk_id": "microsoft_fy24_fy25#p4", "filename": "microsoft_fy24_fy25.md", "company": "Microsoft", "verdict": "irrelevant", "explanation": "Odrzucono: dotyczy konsoli Xbox i Activision, inna spółka."},
                        {"index": 4, "chunk_id": "amazon_fy24_fy25#p2", "filename": "amazon_fy24_fy25.md", "company": "Amazon", "verdict": "irrelevant", "explanation": "Odrzucono: dotyczy usług chmurowych Amazon Luna, brak danych o NVIDIA."},
                    ],
                },
            ),
            ("node_start", {"node": "rewrite_query", "edge_from": "grade_documents", "iteration": 1}),
            (
                "step",
                {
                    "node": "rewrite_query",
                    "latency_ms": 820.0,
                    "total_ms": 2950.0,
                    "retry_count": 1,
                    "new_question": "NVIDIA Gaming segment revenue GeForce RTX Q3 FY2025 10-Q",
                    "rewrite_info": {
                        "original_query": "Ile NVIDIA zarobiła na kartach do grania w Q3 FY2025?",
                        "failed_query": "Ile NVIDIA zarobiła na kartach do grania w Q3 FY2025?",
                        "new_query": "NVIDIA Gaming segment revenue GeForce RTX Q3 FY2025 10-Q",
                        "retry_number": 1,
                    },
                },
            ),
            ("node_start", {"node": "retrieve", "edge_from": "rewrite_query", "iteration": 1}),
            (
                "step",
                {
                    "node": "retrieve",
                    "latency_ms": 430.0,
                    "total_ms": 3380.0,
                    "documents_count": 10,
                    "candidates": [
                        {"index": 1, "chunk_id": "nvidia_q3_fy25_10q#t2", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "is_table": True, "preview": "| Revenue by Market | Q3 FY25 | Q3 FY24 |\n| Gaming | $3,279M | $2,856M |\n| Data Center | $30,771M | $14,514M |"},
                        {"index": 2, "chunk_id": "nvidia_fy25#p1", "filename": "nvidia_fy25.md", "company": "NVIDIA", "is_table": False, "preview": "Gaming revenue was $3.28 billion, up 15% from a year ago driven by GeForce RTX 40 Series GPUs."},
                    ],
                },
            ),
            ("node_start", {"node": "local_rerank", "edge_from": "retrieve", "iteration": 1}),
            (
                "step",
                {
                    "node": "local_rerank",
                    "latency_ms": 760.0,
                    "total_ms": 4140.0,
                    "gpu_device": "NVIDIA GeForce RTX 2050 (FP16)",
                    "vram_mb": 431.2,
                    "ranked_candidates": [
                        {"rank": 1, "chunk_id": "nvidia_q3_fy25_10q#t2", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "score": 0.991, "kept": True, "is_table": True, "preview": "| Revenue by Market | Q3 FY25 | Q3 FY24 |\n| Gaming | $3,279M | $2,856M |"},
                        {"rank": 2, "chunk_id": "nvidia_fy25#p1", "filename": "nvidia_fy25.md", "company": "NVIDIA", "score": 0.978, "kept": True, "is_table": False, "preview": "Gaming revenue was $3.28 billion, up 15% from a year ago..."},
                    ],
                },
            ),
            ("node_start", {"node": "grade_documents", "edge_from": "local_rerank", "iteration": 1}),
            (
                "step",
                {
                    "node": "grade_documents",
                    "latency_ms": 590.0,
                    "total_ms": 4730.0,
                    "accepted_count": 2,
                    "rejected_count": 0,
                    "graded_verdicts": [
                        {"index": 1, "chunk_id": "nvidia_q3_fy25_10q#t2", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "verdict": "relevant", "explanation": "Precyzyjna tabela przychodów segmentu Gaming ($3,279 mln) za Q3 FY25."},
                        {"index": 2, "chunk_id": "nvidia_fy25#p1", "filename": "nvidia_fy25.md", "company": "NVIDIA", "verdict": "relevant", "explanation": "Szczegółowy opis dynamiki segmentu Gaming napędzanej układami GeForce RTX 40 Series."},
                    ],
                },
            ),
            ("node_start", {"node": "generate", "edge_from": "grade_documents", "iteration": 1}),
            ("token", {"node": "generate", "delta": "W trzecim kwartale roku obrotowego 2025 (Q3 FY2025) segment **Gaming** (kart graficznych dla graczy) spółki **NVIDIA** wygenerował:\n\n"}),
            ("token", {"node": "generate", "delta": "### Wyniki Segmentu Gaming:\n"}),
            ("token", {"node": "generate", "delta": "- **Przychody w Q3 FY2025**: **$3,279 mln** (~**$3.28 mld**)\n"}),
            ("token", {"node": "generate", "delta": "- **Przychody w Q3 FY2024**: $2,856 mln\n"}),
            ("token", {"node": "generate", "delta": "- **Dynamika wzrostu**: **+15% rok do roku** (oraz +14% kwartał do kwartału)\n\n"}),
            ("token", {"node": "generate", "delta": "Głównym motorem wzrostu była wysoka sprzedaż procesorów graficznych z rodziny **GeForce RTX z serii 40** (architektura Ada Lovelace) zarówno w wersji desktopowej, jak i dla laptopów gamingowych z akceleracją AI DLSS 3.5."}),
            (
                "step",
                {
                    "node": "generate",
                    "latency_ms": 1780.0,
                    "total_ms": 6510.0,
                    "generation": (
                        "W trzecim kwartale roku obrotowego 2025 (Q3 FY2025) segment **Gaming** (kart graficznych dla graczy) spółki **NVIDIA** wygenerował:\n\n"
                        "### Wyniki Segmentu Gaming:\n"
                        "- **Przychody w Q3 FY2025**: **$3,279 mln** (~**$3.28 mld**)\n"
                        "- **Przychody w Q3 FY2024**: $2,856 mln\n"
                        "- **Dynamika wzrostu**: **+15% rok do roku** (oraz +14% kwartał do kwartału)\n\n"
                        "Głównym motorem wzrostu była wysoka sprzedaż procesorów graficznych z rodziny **GeForce RTX z serii 40** (architektura Ada Lovelace) zarówno w wersji desktopowej, jak i dla laptopów gamingowych z akceleracją AI DLSS 3.5."
                    ),
                    "citations": [
                        {"chunk_id": "nvidia_q3_fy25_10q#t2", "filename": "nvidia_q3_fy25_10q.md", "company": "NVIDIA", "is_table": True, "snippet": "| Revenue by Market | Q3 FY25 | Q3 FY24 |\n| Gaming | $3,279M | $2,856M |"},
                        {"chunk_id": "nvidia_fy25#p1", "filename": "nvidia_fy25.md", "company": "NVIDIA", "is_table": False, "snippet": "Gaming revenue was $3.28 billion, up 15% from a year ago driven by GeForce RTX 40 Series GPUs."},
                    ],
                },
            ),
            ("node_start", {"node": "hallucination_check", "edge_from": "generate", "iteration": 1}),
            (
                "step",
                {
                    "node": "hallucination_check",
                    "latency_ms": 710.0,
                    "total_ms": 7220.0,
                    "hallucination_grade": "grounded",
                    "answer_grade": "useful",
                },
            ),
            ("complete", {"status": "done", "thread_id": "replay-self-correction", "total_time_ms": 7220.0}),
        ],
    },
}


async def stream_replay_events(scenario_id: str, tempo: float = 1.0) -> AsyncGenerator[str, None]:
    """Generuje strumień SSE odtwarzający zapisany przebieg z regulowanym tempem demo (1x, 2x, 4x)."""
    scenario = REPLAY_SCENARIOS.get(scenario_id)
    if not scenario:
        yield f"event: error\ndata: {json.dumps({'code': 'scenario_not_found', 'message': f'Nie odnaleziono scenariusza: {scenario_id}'})}\n\n"
        return

    tempo = max(0.2, min(tempo, 10.0))
    events = scenario["events"]

    for event_type, payload in events:
        if event_type == "token":
            delay = 0.04 / tempo
        elif event_type == "node_start":
            delay = 0.15 / tempo
        elif event_type == "step":
            node = payload.get("node")
            if node == "local_rerank":
                delay = 0.8 / tempo
            elif node in ("grade_documents", "rewrite_query"):
                delay = 0.5 / tempo
            else:
                delay = 0.25 / tempo
        else:
            delay = 0.05 / tempo

        await asyncio.sleep(delay)
        yield f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
