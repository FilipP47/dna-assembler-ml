# syntax=docker/dockerfile:1.7
#
# Build (CPU, default):  docker build -t dna-assembler-ml .
# Build (GPU wheels):     docker build --build-arg DEVICE=cuda -t dna-assembler-ml:cuda .
# Run (CPU):   docker run --rm -v "$PWD/work:/work" dna-assembler-ml <cmd>
# Run (GPU):   docker run --rm --gpus all -v "$PWD/work:/work" dna-assembler-ml:cuda <cmd>
#              (needs the NVIDIA Container Toolkit on the host; the cu128
#              torch wheel bundles its own CUDA runtime, so no CUDA base
#              image is required here)
#
# Pinned versions: Rust 1.91, Python 3.11.14 (must match ml/.python-version
# exactly, or uv tries to download a managed interpreter at runtime), Mash
# 2.3, PyTorch per ml/uv.lock (`uv sync --frozen` refuses to silently
# re-resolve if the lock file and pyproject.toml drift apart). See
# README.md for which exact wheel versions that resolves to for the
# cpu/cu128 groups.
#
# Rust must be >=1.85 (edition2024 support, pulled in transitively via
# clap_lex); 1.91 matches what's used for local development.
#
# Data and results are meant to live on a bind-mounted volume
# (-v $PWD/work:/work), not baked into the image.

FROM rust:1.91-slim-bookworm AS rust-builder
WORKDIR /build
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM python:3.11.14-slim-bookworm AS runtime
ARG DEVICE=cpu
ARG MASH_VERSION=2.3

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://github.com/marbl/Mash/releases/download/v${MASH_VERSION}/mash-Linux64-v${MASH_VERSION}.tar" -O /tmp/mash.tar \
    && tar -xf /tmp/mash.tar -C /tmp \
    && install -m 0755 "/tmp/mash-Linux64-v${MASH_VERSION}/mash" /usr/local/bin/mash \
    && rm -rf /tmp/mash*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY --from=rust-builder /build/target/release/dna-assembler ./target/release/dna-assembler
COPY ml/pyproject.toml ml/uv.lock ./ml/
RUN cd ml && \
    if [ "$DEVICE" = "cuda" ]; then \
        uv sync --frozen --no-group cpu --group cu128; \
    else \
        uv sync --frozen; \
    fi

COPY . .

ENV ROOT=/app
ENV DEVICE=${DEVICE}
ENV UV_PROJECT_ENVIRONMENT=/app/ml/.venv
ENV PYTHONUNBUFFERED=1
# Without this, `uv run` prefers its own managed CPython over the system
# one the .venv above was already built with, and re-downloads + rebuilds
# the venv on first run inside the container (needs network, ~30s wasted).
ENV UV_PYTHON_PREFERENCE=only-system

VOLUME ["/work"]
ENTRYPOINT ["/bin/bash"]
