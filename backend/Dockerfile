FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY backend/pyproject.toml backend/uv.lock /app/
RUN uv sync --frozen --no-dev
RUN uvx playwright install chromium --with-deps

COPY backend/ /app/
COPY graph_app/ /app/graph_app/

EXPOSE 5001

CMD ["uv", "run", "python", "server.py"]
