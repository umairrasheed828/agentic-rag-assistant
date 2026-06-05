FROM python:3.11-slim

# bring in the uv binary from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# install dependencies first (this layer is cached unless deps change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# then copy the application code
COPY src ./src

# put the virtualenv's tools on PATH so we can call uvicorn directly
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]