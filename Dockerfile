# ==============================================================================
# Phase 5 — Dockerfile: Secure Multi-Stage Production Build (OWASP Compliant)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependency Assembly & compilation
# ------------------------------------------------------------------------------
FROM python:3.9-slim AS builder

WORKDIR /app

# Install compilation headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages into isolated directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Final High-Security Runtime Runner
# ------------------------------------------------------------------------------
FROM python:3.9-slim AS runner

WORKDIR /app

# Install shared libs needed for Scapy raw socket bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap0.8 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy assembled user packages from Stage 1 builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Create isolated, non-privileged system group & user (no-shell, no-home)
RUN groupadd -r nids && useradd -r -g nids -d /app -s /sbin/nologin nids

# Copy application files
COPY . .

# Set strict folder ownership
RUN chown -R nids:nids /app && \
    chmod -R 755 /app

# Switch context to the secure sandbox user
USER nids

EXPOSE 8501

# Container health probe checking Streamlit endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
