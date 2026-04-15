FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV DISABLE_SAFETENSORS_CONVERSION=1
ENV CODLLM_OUTPUT_DIR=/app/runs
ENV CODLLM_DATA_PROCESSED_DIR=/app/data/processed

COPY uv.lock pyproject.toml README.md ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY data/ data/

RUN uv sync --frozen --no-dev

RUN mkdir -p /app/runs /app/data/processed

VOLUME ["/app/runs", "/app/data/processed"]

CMD ["uv", "run", "python", "-m", "codllm.training"]
