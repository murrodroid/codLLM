FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY uv.lock pyproject.toml README.md ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "-m", "codllm.train"]
