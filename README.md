<!--
═══════════════════════════════════════════════════════════════════════════════
  LLM CONTEXT BLOCK — NOT FOR HUMAN READING
  This section is intentionally placed before the human-readable documentation.
  It is structured for fast, unambiguous ingestion by language models operating
  in code analysis, refactoring, or pair-programming contexts.

  WHY THIS EXISTS:
  LLMs benefit from a dense, context-first summary that eliminates the need to
  parse narrative prose before understanding the system. This block makes the
  project's contracts, invariants and architectural decisions immediately
  available at the top of the context window, reducing inference errors caused
  by incomplete or late-arriving context.

  WHY IT IS SEPARATED FROM THE HUMAN SECTION:
  Human readers benefit from progressive disclosure — architecture first, then
  details. LLMs benefit from the opposite: maximum density upfront, before any
  prose. Mixing both would degrade readability for humans and force LLMs to
  filter noise. The HTML comment wrapper ensures this block is invisible in
  rendered Markdown while remaining fully readable in raw source (the context
  in which LLMs typically operate).

  ─────────────────────────────────────────────────────────────────────────────
  PROJECT IDENTITY
  ─────────────────────────────────────────────────────────────────────────────
  Name:       WhispeRust
  Binary:     whisperust
  Language:   Rust (edition 2021, MSRV 1.78)
  Runtime:    tokio (multi-thread)
  Role:       Async RabbitMQ worker — consumes audio jobs, transcribes via
              whisper.cpp (GGML), publishes results back to RabbitMQ.
  Origin:     Rust rewrite of a Go + Python stack. Messaging contracts
              are protocol-identical to the original implementation.

  ─────────────────────────────────────────────────────────────────────────────
  CRATE DEPENDENCY MAP (purpose-annotated)
  ─────────────────────────────────────────────────────────────────────────────
  tokio 1                  — async multi-thread runtime
  lapin 2.3                — AMQP 0-9-1 client
  deadpool-lapin 0.12      — connection pool over lapin (10 retries, 5 s interval)
  serde 1 + serde_json 1   — JSON serialisation / deserialisation
  symphonia 0.5            — audio decode: MP3, WAV, FLAC, OGG/Vorbis, AAC, M4A
  ogg 0.8 + opus 0.3       — OGG/Opus decode via system libopus (bypasses Symphonia)
  rubato 0.15              — SincFixedIn resampler → mono f32 @ 16 000 Hz
  whisper-rs 0.13          — unsafe bindings to whisper.cpp GGML C library
  tracing 0.1              — structured, async-aware logging
  tracing-subscriber 0.3   — RUST_LOG env filter subscriber

  ─────────────────────────────────────────────────────────────────────────────
  MODULE TREE (path → primary type / function exported)
  ─────────────────────────────────────────────────────────────────────────────
  src/main.rs              — #[tokio::main]; tracing_subscriber init; app::run()
  src/app.rs               — async fn run() → Result<(), AppError>; full startup/teardown
  src/config.rs            — struct Config; Config::load() from env vars; Config::model_path()
  src/retry.rs             — MAX_RETRIES: i32 = 2; enum RetryDecision; RetryPolicy::decide()
  src/metrics.rs           — struct Metrics (AtomicU64×4, AtomicI64×1); Arc<Metrics>
  src/shutdown.rs          — fn new_pair() → (ShutdownHandle, ShutdownSignal); wait_for_os_signal()
  src/model/request.rs     — struct TranscriptionRequest { attachment_id, audio_file_path,
                               language, import_batch_id, retry_count }
  src/model/result.rs      — struct TranscriptionResult { attachment_id, texto, duration,
                               model, success, import_batch_id, error_message }
                             TranscriptionResult::success(…) / ::failure(…)
  src/audio/decoder.rs     — fn decode(path) → Result<DecodedAudio, DecodeError>
                             OGG/Opus path: is_ogg_opus() → decode_ogg_opus() (libopus)
                             All other formats: Symphonia probe/decode loop
  src/audio/resampler.rs   — fn to_mono_16k(decoded: &DecodedAudio) → Vec<f32>  (rubato)
  src/audio/pipeline.rs    — struct AudioLimits; struct ProcessedAudio { samples, duration_secs }
                             fn process(path, limits) → Result<ProcessedAudio, AudioError>
                             Validation order: exists+size → extension → decode → duration → resample
  src/whisper/model.rs     — struct WhisperModel { context: WhisperContext, name: String }
                             unsafe impl Send + Sync; fn load(path, name) → Result<Self>
  src/whisper/engine.rs    — #[derive(Clone)] struct WhisperEngine { model: Arc<WhisperModel> }
                             fn transcribe(&self, samples, language) → Result<TranscriptionOutput>
                             BeamSearch { beam_size: 5 }; no_speech_thold = 0.6
  src/messaging/rabbit.rs  — pub type Pool = deadpool_lapin::Pool; build_pool(url, max_conn)
                             Constants: exchange/queue/routing-key names; RETRY_TTL_MS = 5000
  src/messaging/consumer.rs — struct Job { request, delivery }; RabbitConsumer::new()
                              into_receiver() → mpsc::Receiver<Job>; NACK invalid JSON (no requeue)
  src/messaging/producer.rs — #[derive(Clone)] RabbitProducer; publish_success / publish_error
                              publish_retry: increments retry_count; sets x-retry-count AMQP header
  src/worker/task.rs       — async fn process(worker_id, job, engine, producer, limits, metrics)
                             enum TaskError { Deterministic, Transient }
                             Deterministic errors → no retry (FileNotFound, UnsupportedFormat,
                             FileTooLarge, DurationExceeded)
  src/worker/pool.rs       — struct WorkerPool; fn run(self, jobs_rx, shutdown_signal)
                             N worker tasks share Arc<Mutex<Receiver>>; capacity = workers×2

  ─────────────────────────────────────────────────────────────────────────────
  AMQP TOPOLOGY (exact names used in code)
  ─────────────────────────────────────────────────────────────────────────────
  Input:
    exchange     = "whisper_exchange"          (direct)
    queue        = "whisper_transcriptions"    (durable)
    routing_key  = "transcription.request"

  Output:
    exchange     = "whisper_results_exchange"  (direct)
    queue        = "whisper_results"           (durable)
    routing_key  = "transcription.result"

  Retry (DLX pattern):
    exchange     = "whisper_retry_exchange"    (direct)
    queue        = "whisper_retry_queue"       (durable, x-message-ttl=5000ms)
    routing_key  = "transcription.retry"
    DLX on expiry → "whisper_exchange"         (re-enters main queue)
    MAX_RETRIES  = 2  (3 total attempts)

  ─────────────────────────────────────────────────────────────────────────────
  MESSAGE CONTRACTS (canonical JSON — fields are domain-specific to the
  original application; adapt field names/types for other use cases)
  ─────────────────────────────────────────────────────────────────────────────
  Request  (inbound):
    { "attachment_id": i64, "audio_file_path": String,
      "language": String|null, "import_batch_id": i64|null, "retry_count": i32 }
    — audio_file_path: absolute path accessible inside the container
    — retry_count: managed by the service; send 0 or omit

  Result  (outbound, success):
    { "attachment_id": i64, "texto": String, "duration": f64,
      "model": String, "success": true, "import_batch_id": i64|null }

  Result  (outbound, failure):
    { "attachment_id": i64, "texto": "", "duration": 0.0,
      "model": String, "success": false, "import_batch_id": i64|null,
      "error_message": String }
    — error_message: serialized only on failure (skip_serializing_if = None)

  ─────────────────────────────────────────────────────────────────────────────
  KEY INVARIANTS (enforce when modifying code)
  ─────────────────────────────────────────────────────────────────────────────
  1. AUDIO_SAMPLE_RATE must equal 16000 — whisper.cpp hard requirement.
  2. WhisperModel is loaded exactly once at startup (expensive). Arc<WhisperModel>
     is cloned into each WhisperEngine instance.
  3. WhisperState is created fresh per transcription call (not reused).
  4. OGG/Opus detection reads first 128 bytes for "OpusHead" magic bytes,
     before file extension check, to correctly handle .ogg containers.
  5. Deterministic audio errors (FileNotFound, UnsupportedFormat, FileTooLarge,
     DurationExceeded) MUST NOT be retried — publish failure result immediately.
  6. RABBITMQ_URL default vhost must use %2F encoding, not a bare slash.
  7. Model file path convention: {MODELS_DIR}/ggml-{WHISPER_MODEL}.bin
  8. entrypoint.sh auto-downloads the model at container start if missing.

  ─────────────────────────────────────────────────────────────────────────────
  DOCKER / BUILD NOTES
  ─────────────────────────────────────────────────────────────────────────────
  Builder stage: rust:1-bookworm + libopus-dev + libclang-dev + cmake
  Runtime stage: debian:bookworm-slim + libopus0 + libgomp1 + libstdc++6
  Dep-cache trick: stub main.rs compiled first, then real src/ copied and built.
  After modifying src/: rebuild with --no-cache to bypass COPY src layer cache.
  Command: docker-compose build --no-cache whisperust

═══════════════════════════════════════════════════════════════════════════════
  END OF LLM CONTEXT BLOCK — HUMAN DOCUMENTATION BEGINS BELOW
═══════════════════════════════════════════════════════════════════════════════
-->

# WhispeRust

Async Rust service that consumes audio transcription jobs from RabbitMQ, decodes the audio, runs inference with [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (GGML models), and publishes the result back to RabbitMQ.

Drop-in replacement for the original Go + Python stack. **Messaging contracts are identical.**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         WhispeRust                           │
│                                                              │
│  RabbitMQ ──► Consumer ──► WorkerPool ──► Producer ──► RabbitMQ
│               │             │    │                          │
│               │          N workers                         │
│               │          ┌──────────────────────────────┐  │
│               │          │  AudioPipeline               │  │
│               │          │  ├─ validate (size/duration) │  │
│               │          │  ├─ decode (Symphonia/Opus)  │  │
│               │          │  └─ resample → mono 16 kHz   │  │
│               │          │                              │  │
│               │          │  WhisperEngine               │  │
│               │          │  └─ whisper.cpp (GGML)       │  │
│               └──────────└──────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Module map

| Module | Responsibility |
|---|---|
| `main.rs` | Entry point, tracing initialisation |
| `app.rs` | Startup / graceful-shutdown orchestration |
| `config.rs` | Environment-variable configuration + validation |
| `retry.rs` | Retry policy (MAX_RETRIES = 2, 5 s delay via DLX) |
| `metrics.rs` | Atomic job counters (received / succeeded / failed / retried / in-flight) |
| `shutdown.rs` | Watch-channel pair + SIGINT/SIGTERM handler |
| `model/` | `TranscriptionRequest` and `TranscriptionResult` structs |
| `audio/` | Decode, resample, validate — full audio pipeline |
| `whisper/` | `WhisperModel` (load once) + `WhisperEngine` (transcribe) |
| `messaging/` | AMQP connection pool, consumer, producer |
| `worker/` | `WorkerPool` dispatcher + per-job `task::process` |

---

## Messaging Contracts

> **Nota de implementación:** Los campos `attachment_id` e `import_batch_id` corresponden al dominio de la aplicación original para la que fue diseñado el servicio. Los contratos de mensajería **pueden y deben** adaptarse a las necesidades de cada caso de uso particular — los nombres de campos, tipos y semántica son completamente modificables siempre que el productor y el consumidor estén de acuerdo. Puede haber coincidencia con el contrato aquí documentado o diferir completamente.

### Input — Transcription Request

Consumed from:
- Exchange: `whisper_exchange` (direct)
- Queue: `whisper_transcriptions`
- Routing key: `transcription.request`

```json
{
  "attachment_id":   123,
  "audio_file_path": "/mnt/audio/voice_001.ogg",
  "language":        "es",
  "import_batch_id": 42,
  "retry_count":     0
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `attachment_id` | `i64` | ✅ | Identifies the asset being transcribed |
| `audio_file_path` | `String` | ✅ | Absolute path to the audio file on disk |
| `language` | `String` | ✗ | ISO 639-1 hint (`"es"`, `"en"`, …). `null` = auto-detect |
| `import_batch_id` | `i64` | ✗ | Passed through unchanged to the result |
| `retry_count` | `i32` | ✗ | Set automatically on retry; start at `0` |

---

### Output — Transcription Result

Published to:
- Exchange: `whisper_results_exchange` (direct)
- Queue: `whisper_results`
- Routing key: `transcription.result`

**Success:**
```json
{
  "attachment_id":   123,
  "texto":           "Hola, ¿cómo estás?",
  "duration":        4.32,
  "model":           "base-q5_1",
  "success":         true,
  "import_batch_id": 42
}
```

**Failure:**
```json
{
  "attachment_id":   123,
  "texto":           "",
  "duration":        0.0,
  "model":           "base-q5_1",
  "success":         false,
  "import_batch_id": 42,
  "error_message":   "file not found: /mnt/audio/voice_001.ogg"
}
```

| Field | Type | Description |
|---|---|---|
| `attachment_id` | `i64` | Echoed from the request |
| `texto` | `String` | Transcription text (empty on failure) |
| `duration` | `f64` | Audio duration in seconds |
| `model` | `String` | Model identifier used |
| `success` | `bool` | Whether transcription succeeded |
| `import_batch_id` | `i64 \| null` | Echoed from the request |
| `error_message` | `String \| null` | Present only on failure |

---

### Retry Topology

```
Main queue ──► [fail, transient] ──► whisper_retry_exchange
                                          │
                                     whisper_retry_queue  (TTL = 5 s)
                                          │  x-dead-letter-exchange
                                          └──► whisper_exchange  (re-queued)
```

- **MAX_RETRIES = 2** (3 total attempts). After the third failure the job is NACKed without requeue and a failure result is published.
- Deterministic errors (file not found, unsupported format, file too large, duration exceeded) are **not retried**.

---

## Configuration

All settings are read from environment variables at startup.

| Variable | Default | Constraint | Description |
|---|---|---|---|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | valid AMQP URL | Use `%2F` for the default vhost |
| `RABBITMQ_QUEUE_NAME` | `whisper_transcriptions` | — | Input queue name |
| `WORKERS_COUNT` | `4` | ≥ 1 | Concurrent transcription workers |
| `WHISPER_MODEL` | `base` | — | Model ID, e.g. `base-q5_1`, `small`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` / `cuda` / `metal` | Inference device |
| `MODELS_DIR` | `/app/models` | — | Directory with GGML `.bin` files |
| `MAX_FILE_SIZE_MB` | `100` | ≥ 1 | Maximum accepted file size |
| `MAX_AUDIO_DURATION_SEC` | `3600` | > 0 | Maximum accepted audio duration |
| `AUDIO_SAMPLE_RATE` | `16000` | must = 16000 | whisper.cpp requires exactly 16 kHz |
| `TMP_DIR` | `/tmp/whisper` | — | Directory for intermediate files |
| `API_HOST` | `0.0.0.0` | — | HTTP API bind address (reserved) |
| `API_PORT` | `8080` | 1–65535 | HTTP API port (reserved) |
| `RUST_LOG` | `whisperust=info,warn` | tracing filter | Log level |

The model file is resolved as: `{MODELS_DIR}/ggml-{WHISPER_MODEL}.bin`

---

## Supported Audio Formats

| Format | Extension | Decoder |
|---|---|---|
| OGG / Opus | `.opus`, `.ogg` (OpusHead) | `ogg` + `opus` crates via `libopus` |
| MP3 | `.mp3` | Symphonia |
| WAV | `.wav` | Symphonia |
| FLAC | `.flac` | Symphonia |
| AAC / M4A | `.aac`, `.m4a` | Symphonia |
| OGG / Vorbis | `.ogg` (non-Opus) | Symphonia |

All formats are resampled to **mono 16 kHz f32** before inference.

---

## Running with Docker Compose

```bash
# First run — builds the image, downloads the model, starts services
docker-compose up --build

# Rebuild after source changes (force-bypass cache)
docker-compose build --no-cache whisperust
docker-compose up
```

The `entrypoint.sh` automatically downloads the GGML model from Hugging Face if it is missing:

```
https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{WHISPER_MODEL}.bin
```

To pre-download manually:
```bash
mkdir -p models
MODEL=base-q5_1
curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin" \
     -o "models/ggml-${MODEL}.bin"
```

---

## Building Locally

**Prerequisites:**
- Rust 1.78+
- `libopus-dev` (for OGG/Opus decoding)
- `libclang-dev`, `cmake` (for whisper.cpp build)

```bash
cd WhispeRust
cargo build --release
# binary: ./target/release/whisperust
```

### Docker Compose example

Copy and paste this as your `docker-compose.yml` at the project root. Assumes the `WhispeRust/` directory is a sibling of `docker-compose.yml` and that models are stored in a `models/` bind mount.

```yaml
services:

  rabbitmq:
    image: rabbitmq:3-management
    container_name: rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"   # management UI → http://localhost:15672
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 10s
      timeout: 5s
      retries: 10

  whisperust:
    build:
      context: ./WhispeRust
      dockerfile: Dockerfile
    container_name: whisperust
    depends_on:
      rabbitmq:
        condition: service_healthy
    volumes:
      - ./models:/app/models   # persists downloaded GGML models
    environment:
      RABBITMQ_URL: amqp://admin:admin@rabbitmq:5672/%2F
      RABBITMQ_QUEUE_NAME: whisper_transcriptions
      WORKERS_COUNT: "4"
      WHISPER_MODEL: base-q5_1
      WHISPER_DEVICE: cpu
      MODELS_DIR: /app/models
      MAX_FILE_SIZE_MB: "100"
      MAX_AUDIO_DURATION_SEC: "3600"
      TMP_DIR: /tmp/whisper
      RUST_LOG: whisperust=info
    restart: unless-stopped
```

> **Note:** Audio files must be accessible **inside the container**. Mount the directory that contains your audio files as an additional volume, e.g. `- /data/audio:/mnt/audio`, and send `audio_file_path` values using the container-side path (`/mnt/audio/...`).

---

## Dependencies

| Crate | Purpose |
|---|---|
| `tokio` | Async runtime |
| `lapin` + `deadpool-lapin` | AMQP 0-9-1 client + connection pool |
| `serde` + `serde_json` | JSON serialisation |
| `symphonia` | Audio decode (MP3, WAV, FLAC, AAC, OGG/Vorbis) |
| `ogg` + `opus` | OGG/Opus decode via `libopus` |
| `rubato` | SincFixedIn resampler → mono 16 kHz |
| `whisper-rs` | whisper.cpp bindings (GGML inference) |
| `tracing` + `tracing-subscriber` | Structured logging |
