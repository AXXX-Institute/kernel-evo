# Build from repo root: docker build -f build/Dockerfile .
# Run with custom problems: mount your task dir and pass --problem-path (see build/README.md).

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

WORKDIR /app

# Git required for pip to install git+https:// deps (gigaevo, kernelbench)
# Redis optional for GigaEvo state
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Install kernel-evo and dependencies from the project
COPY pyproject.toml README.md ./
COPY src ./src

# Optional: pass GitHub token for private deps (gigaevo, kernelbench).
#   docker build --secret id=github_token,env=GITHUB_TOKEN -f build/Dockerfile .
# Token is not stored in image layers (BuildKit secret).
RUN --mount=type=secret,id=github_token \
    if [ -f /run/secrets/github_token ] && [ -s /run/secrets/github_token ]; then \
      printf '%s\n' '#!/bin/sh' 'echo "username=oauth2"' 'echo "password=$(cat /run/secrets/github_token)"' > /tmp/git-cred-helper && \
      chmod +x /tmp/git-cred-helper && \
      git config --global credential.helper "/tmp/git-cred-helper"; \
    fi && \
    pip install --no-cache-dir -e . --ignore-requires-python

# Default: show CLI help
CMD ["kernel-evo", "--help"]
