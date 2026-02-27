FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        clang \
        libclang-dev \
        pkg-config \
        libopus-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.lock* ./

RUN mkdir -p src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release --bin ruistper || true \
    && rm -f  src/main.rs \
              target/release/ruistper \
              target/release/deps/ruistper-* \
    && rm -rf target/release/.fingerprint/ruistper-*

COPY src ./src
RUN cargo build --release --bin ruistper

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
        libopus0 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/ruistper /usr/local/bin/ruistper

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /app/models /tmp/shared_audio

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
    RUST_LOG=ruistper=info,warn

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/local/bin/ruistper"]
