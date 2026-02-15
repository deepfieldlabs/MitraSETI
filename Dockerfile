# =============================================================================
# astroSETI — Multi-stage Docker Build
# Stage 1: Build Rust core
# Stage 2: Python runtime with compiled extension
# =============================================================================

# ---------------------
# Stage 1: Rust Builder
# ---------------------
FROM rust:1.75-bookworm AS rust-builder

WORKDIR /build

# Copy Rust project files
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

# Build the Rust library in release mode
RUN cargo build --release

# ---------------------
# Stage 2: Python App
# ---------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL maintainer="Saman Tabatabaeian <saman@astroseti.dev>"
LABEL description="astroSETI — Intelligent SETI Signal Analysis"
LABEL org.opencontainers.image.source="https://github.com/SamanTabworlds/astroSETI"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for maturin build
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install Python dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Build the Rust extension via maturin
RUN maturin develop --release

# Copy pre-built Rust artifacts from builder stage
COPY --from=rust-builder /build/target/release/*.so /app/lib/ 2>/dev/null || true
COPY --from=rust-builder /build/target/release/*.dylib /app/lib/ 2>/dev/null || true

# Create non-root user
RUN useradd --create-home --shell /bin/bash astroseti
RUN chown -R astroseti:astroseti /app
USER astroseti

# Expose ports: 8000 for API, 8080 for web UI
EXPOSE 8000
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
