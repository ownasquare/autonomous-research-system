# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.17 AS uv-bin

FROM python:3.11-slim-bookworm AS builder
COPY --from=uv-bin /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.11-slim-bookworm AS runtime
RUN groupadd --gid 10001 researchdesk \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/researchdesk researchdesk \
    && install -d -m 0700 -o 10001 -g 10001 /app/data \
    && chown -R 10001:10001 /app /home/researchdesk
WORKDIR /app
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 .streamlit ./.streamlit
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RESEARCH_DATA_DIR=/app/data \
    RESEARCH_MODE=demo
USER 10001:10001
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]
CMD ["streamlit", "run", "src/research_system/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
