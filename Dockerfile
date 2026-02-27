# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WhispeRust — Multi-stage Dockerfile                                    ║
# ║                                                                         ║
# ║  Stage 1 (builder) : Rust toolchain + C++ build tools                  ║
# ║    · cmake + build-essential: compilan whisper.cpp (via whisper-rs-sys) ║
# ║    · clang + libclang-dev   : requeridos por bindgen (FFI bindings)     ║
# ║                                                                         ║
# ║  Stage 2 (runtime) : debian:bookworm-slim                               ║
# ║    · Solo el binario + libstdc++ + libgomp + ca-certificates            ║
# ║    · Sin Rust, sin Python, sin ffmpeg, sin herramientas de compilación  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM rust:1-bookworm AS builder

# Toolchain C/C++ necesario para compilar whisper.cpp (incluido en whisper-rs-sys).
# libclang-dev + clang son requeridos por bindgen para generar los bindings FFI.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        clang \
        libclang-dev \
        pkg-config \
        libopus-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── Capa de caché de dependencias ─────────────────────────────────────────────
# Se copia SOLO el manifiesto para que Docker pueda cachear la compilación
# completa de dependencias (incluyendo el costoso build de whisper.cpp en C++)
# de forma independiente a los cambios en el código fuente.
# Esta capa sólo se invalida cuando cambia Cargo.toml o Cargo.lock.
COPY Cargo.toml Cargo.lock* ./

# Compilar un stub mínimo que fuerza a cargo a descargar y compilar todos los
# crates externos (tokio, lapin, symphonia, whisper-rs + whisper.cpp, etc.).
# El "|| true" absorbe el error de enlazado esperado por el stub vacío.
# Luego se borran sólo los artefactos del stub para que el código real
# se compile desde cero en el siguiente paso (evitando binarios corruptos).
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release --bin whisperust || true \
    && rm -f  src/main.rs \
              target/release/whisperust \
              target/release/deps/whisperust-* \
    && rm -rf target/release/.fingerprint/whisperust-*

# ── Compilación real ──────────────────────────────────────────────────────────
COPY src ./src
RUN cargo build --release --bin whisperust

# ── Stage 2: Runtime mínimo ───────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# libstdc++6      — runtime C++ enlazado por los objetos de whisper.cpp
# libgomp1        — runtime OpenMP (whisper.cpp usa -fopenmp para decode multi-hilo)
# ca-certificates — certificados TLS para conexiones AMQP con RabbitMQ
# curl            — descarga automática del modelo GGML en el entrypoint
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
        libopus0 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Único artefacto copiado desde el builder: el binario estático
COPY --from=builder /build/target/release/whisperust /usr/local/bin/whisperust

# Entrypoint: descarga el modelo GGML si no está presente, luego exec el binario
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Directorios requeridos por el servicio.
# Los modelos GGML se montan desde el host via bind mount (ver docker-compose.yml).
RUN mkdir -p /app/models /tmp/shared_audio

# Valores por defecto — todos sobreescribibles desde docker-compose.yml
ENV WHISPER_MODEL=base.q5_0 \
    WHISPER_DEVICE=cpu \
    MODELS_DIR=/app/models \
    MAX_FILE_SIZE_MB=25 \
    MAX_AUDIO_DURATION_SEC=300 \
    AUDIO_SAMPLE_RATE=16000 \
    TMP_DIR=/tmp/shared_audio \
    WORKERS_COUNT=1 \
    RABBITMQ_URL=amqp://guest:guest@localhost:5672/ \
    RABBITMQ_QUEUE_NAME=whisper_transcriptions \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    RUST_LOG=whisperust=info,warn

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/local/bin/whisperust"]
