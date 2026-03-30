# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Install FFmpeg (required for video processing) ────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy app code ─────────────────────────────────────────────────────────────
COPY . .

# ── Create data directories ───────────────────────────────────────────────────
RUN mkdir -p /app/uploads /app/outputs

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
