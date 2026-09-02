"""Moduł pomocniczy do bezpiecznego wykonywania asynchronicznych zadań."""

import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Bezpiecznie uruchamia asynchroniczną korutynę zarówno w kontekście synchronicznym,
    jak i w wątkach z już działającą pętlą zdarzeń asyncio.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Jesteśmy wewnątrz działającej pętli zdarzeń -> uruchomienie w osobnym wątku roboczym
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
