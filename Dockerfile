# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies first (less likely to change)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set PATH for uv and potentially cargo
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Copy only dependency definition files
COPY requirements.txt .
COPY lightrag/api/requirements.txt ./lightrag/api/
COPY setup.py .
COPY lightrag/__init__.py ./lightrag/

# Compile and install dependencies (caches well if definition files don't change)
RUN uv pip compile requirements.txt -o requirements.lock
RUN uv pip compile lightrag/api/requirements.txt -o api-requirements.lock
RUN uv pip install --system --no-cache-dir -r requirements.lock
RUN uv pip install --system --no-cache-dir -r api-requirements.lock
RUN uv run opentelemetry-bootstrap -a requirements | uv pip install --system --no-cache-dir --requirement -

# Copy application code (changes here will invalidate subsequent layers)
COPY ./lightrag ./lightrag/

# Install the application itself
RUN uv pip install --system -e .

# Final stage
FROM python:3.11-slim AS final

WORKDIR /app

# Copy Python installation with all installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
# Copy the uv executable from the builder stage
COPY --from=builder /root/.local/bin/uv /usr/local/bin/uv

# Copy application code
COPY ./lightrag ./lightrag/
COPY setup.py .

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Docker data directories
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

# OpenTelemetry configuration for Langfuse
ENV LANGFUSE_PUBLIC_KEY="pk-lf-417f6001-b824-405a-bc02-7ef6ff3771a9"
ENV LANGFUSE_SECRET_KEY="sk-lf-e64e651e-1899-49cd-934b-0937ae5bf85d"
ENV LANGFUSE_HOST="https://cloud.langfuse.com"
ENV OTEL_SERVICE_NAME="lightrag"
ENV OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
ENV OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://cloud.langfuse.com/api/public/otel/v1/traces"
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="false"
ENV OTEL_TRACES_EXPORTER="otlp"
ENV OTEL_METRICS_EXPORTER="none"
ENV OTEL_LOGS_EXPORTER="none"
ENV OTEL_PYTHON_EXCLUDED_URLS="health"

# Set OpenTelemetry headers directly
ENV OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic cGstbGYtNDE3ZjYwMDEtYjgyNC00MDVhLWJjMDItN2VmNmZmMzc3MWE5OnNrLWxmLWU2NGU2NTFlLTE4OTktNDljZC05MzRiLTA5MzdhZTViZjg1ZA=="
ENV OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Basic cGstbGYtNDE3ZjYwMDEtYjgyNC00MDVhLWJjMDItN2VmNmZmMzc3MWE5OnNrLWxmLWU2NGU2NTFlLTE4OTktNDljZC05MzRiLTA5MzdhZTViZjg1ZA=="

# Expose the default port
EXPOSE 9621

# Set entrypoint using uv run as recommended by OpenTelemetry docs
ENTRYPOINT ["uv", "run", "opentelemetry-instrument", "python", "lightrag/api/lightrag_server.py"]
