FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH=/app/.venv/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib \
    PREFECT_HOME=/app/state/prefect \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true

RUN useradd --create-home --uid 10001 neuroops \
    && pip install --no-cache-dir uv==0.11.33
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    && rm -rf /tmp/uv-cache
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY prefect.yaml ./prefect.yaml
RUN uv sync --frozen --no-dev \
    && mkdir -p /app/data/uploads /app/data/processed /app/artifacts /app/state \
    && chown -R neuroops:neuroops /app
USER neuroops
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=8 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)" || exit 1
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

FROM base AS eeg
USER root
RUN uv sync --frozen --no-dev --extra eeg && rm -rf /tmp/uv-cache && chown -R neuroops:neuroops /app
USER neuroops

FROM eeg AS deep-learning
USER root
RUN uv sync --frozen --no-dev --extra eeg --extra deep-learning && rm -rf /tmp/uv-cache && chown -R neuroops:neuroops /app
USER neuroops

FROM eeg AS production
USER root
RUN uv sync --frozen --no-dev --extra eeg --extra postgres && rm -rf /tmp/uv-cache && chown -R neuroops:neuroops /app
USER neuroops
