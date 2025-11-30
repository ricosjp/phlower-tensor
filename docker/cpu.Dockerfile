# 3.10, 3.11, 3.12
ARG USE_PYTHON_VERSION=3.12

# ----- 1. uv builder stage
FROM ubuntu:20.04 AS uv_builder
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates make
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh


# ----- 2. builder stage
FROM python:${USE_PYTHON_VERSION}-slim AS builder
ARG USE_PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=uv_builder /root/.local/bin/uv /root/.local/bin/uvx /bin/

# Copy the project into the image
COPY pyproject.toml /workspace/
COPY Makefile /workspace/
COPY README.md /workspace/
COPY src /workspace/src

WORKDIR /workspace
RUN uv sync --extra cpu --group dev


# ----- 3. final stage
FROM python:${USE_PYTHON_VERSION}-slim
ENV DEBIAN_FRONTEND=noninteractive
ARG USE_PYTHON_VERSION
WORKDIR /workspace

# Enable to use uv
COPY --from=uv_builder /root/.local/bin/uv /root/.local/bin/uvx /bin/
COPY --from=uv_builder /usr/bin/make /usr/bin/
COPY --from=builder /workspace/.venv /workspace/.venv

# UV configuration
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
ENV UV_NO_SYNC=true
ENV UV_PYTHON_INSTALL_DIR=/workspace

RUN chmod +x ./.venv/bin/activate
