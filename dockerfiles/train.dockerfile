FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV DISABLE_SAFETENSORS_CONVERSION=1

COPY uv.lock pyproject.toml README.md ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY data/ data/

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "-m", "codllm.train"]
