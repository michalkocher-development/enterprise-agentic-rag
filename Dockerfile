# ── Enterprise Agentic RAG Platform: Dockerfile dla Hugging Face Spaces ──
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    DEVICE=cpu

# Podstawowe narzędzia systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Konfiguracja użytkownika bez uprawnień roota (wymóg bezpieczeństwa Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# 1. Instalacja PyTorch w lekkiej wersji CPU (redukcja rozmiaru obrazu o ~3.5 GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Instalacja zależności produkcyjnych
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-cache wag modelu Cross-Encoder (BAAI/bge-reranker-base)
# Dzięki temu Hugging Face Space startuje natychmiastowo bez opóźnień sieciowych
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base', device='cpu')"

# 4. Kopiowanie kodu aplikacji, bazy wiedzy i cache wektorowego
COPY --chown=user:user . .

# Port domyślny Hugging Face Spaces
EXPOSE 7860

# Uruchomienie serwera FastAPI
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
