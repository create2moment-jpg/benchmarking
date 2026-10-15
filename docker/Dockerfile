# Task environment for IDE agents (VS Code Dev Container). Contains every tool the benchmark
# tasks need; the agent works in /workspace (the session folder) as the non-root user `dev`,
# who may install more packages (pip into /opt/venv, or `sudo apt-get`) inside the container.
# A second stage, `eval`, adds the benchmark's own dependencies; finish.sh scores sessions in
# it, so solutions are evaluated in a clean copy of exactly this environment.
# Build: docker build --target tasks -t elb-tasks docker/   (setup.sh does this automatically)
FROM python:3.12-slim-bookworm AS tasks

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG WITH_TORCH=1
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      g++ cmake make ninja-build git curl ca-certificates less procps sudo \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-tasks.txt /tmp/requirements-tasks.txt
RUN python -m venv /opt/venv \
 && pip install --no-cache-dir -r /tmp/requirements-tasks.txt \
 && if [ "$WITH_TORCH" = "1" ]; then \
      pip install --no-cache-dir torch --index-url "$TORCH_INDEX" \
      || pip install --no-cache-dir torch \
      || echo "WARNING: PyTorch not installed; the PyTorch task will be skipped"; fi

# Non-root user (Dev Containers remaps its UID to yours so files in the session stay yours).
RUN useradd --create-home --shell /bin/bash --uid 1000 dev \
 && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
 && chmod 0440 /etc/sudoers.d/dev \
 && chown -R dev:dev /opt/venv \
 && mkdir -p /workspace && chown dev:dev /workspace

USER dev
WORKDIR /workspace
CMD ["sleep", "infinity"]

# ---------------------------------------------------------------------------------------------
FROM tasks AS eval
COPY requirements-bench.txt /tmp/requirements-bench.txt
RUN pip install --no-cache-dir -r /tmp/requirements-bench.txt
