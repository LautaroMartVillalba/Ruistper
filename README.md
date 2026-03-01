# Ruistper

---

## 🤖 LLM Context

> This section is structured for fast, unambiguous ingestion by language models. Maximum information density, minimum narrative prose. If you are a human, jump to [👥 Human Documentation](#-human-documentation).

---

### Project Identity

| Field | Value |
|---|---|
| **Name** | Ruistper |
| **Binary** | `ruistper` |
| **Language** | Rust, edition 2021, MSRV 1.78 |
| **Async runtime** | `tokio` multi-thread |
| **Role** | RabbitMQ worker — consumes audio jobs → transcribes via whisper.cpp (GGML) → publishes result back to RabbitMQ |
| **Origin** | Rust rewrite of a Go + Python stack; messaging contracts are protocol-identical to the original |

---

### Crate Dependency Map

| Crate | Version | Purpose |
|---|---|---|
| `tokio` | 1 | Async multi-thread runtime (`rt-multi-thread`, `macros`, `sync`, `fs`, `signal`, `time`) |
| `lapin` | 2.3 | AMQP 0-9-1 client |
| `deadpool-lapin` | 0.12 | Connection pool over lapin |
| `futures-util` | 0.3 | `StreamExt` on consumer delivery stream |
| `serde` + `serde_json` | 1 | JSON serialisation / deserialisation |
| `symphonia` | 0.5 | Audio decode: MP3, WAV, FLAC, OGG/Vorbis, AAC, M4A |
| `ogg` | 0.8 | OGG container demux for Opus streams (bypasses Symphonia) |
| `opus` | 0.3 | OGG/Opus decode via FFI to system `libopus` |
| `rubato` | 0.15 | `SincFixedIn` resampler → mono f32 @ 16 000 Hz |
| `hound` | 3 | WAV writing for optional intermediate files |
| `whisper-rs` | 0.13 | Unsafe bindings to whisper.cpp GGML C library |
| `tracing` | 0.1 | Structured, async-aware logging |
| `tracing-subscriber` | 0.3 | `RUST_LOG` env-filter subscriber |

---

### Module Tree

```
src/main.rs
  #[tokio::main]; tracing_subscriber init; app::run()

src/app.rs
  async fn run() → Result<(), AppError>
  Startup sequence: Config → tmp_dir → Metrics → shutdown pair
    → RabbitMQ pool (workers+2) → WhisperModel (blocking, once)
    → WhisperEngine → RabbitProducer → RabbitConsumer → WorkerPool

src/config.rs
  struct Config { rabbitmq_url, workers_count, whisper_model,
                  whisper_device, models_dir, max_file_size_mb,
                  max_audio_duration_sec, audio_sample_rate,
                  tmp_dir, api_host, api_port }
  Config::load() → reads env vars, validates
  Config::model_path() → PathBuf  ({models_dir}/ggml-{whisper_model}.bin)
  Config::max_file_size_bytes() → u64
  Config::log_summary()

src/retry.rs
  const MAX_RETRIES: i32 = 2   // 3 total attempts
  enum RetryDecision { Retry, GiveUp }
  RetryPolicy::decide(retry_count) → RetryDecision

src/metrics.rs
  struct Metrics {
    received:  AtomicU64,
    succeeded: AtomicU64,
    failed:    AtomicU64,
    retried:   AtomicU64,
    in_flight: AtomicI64,
  }
  Arc<Metrics> shared across workers

src/shutdown.rs
  fn new_pair() → (ShutdownHandle, ShutdownSignal)
  fn wait_for_os_signal()   // SIGINT + SIGTERM

src/model/request.rs
  struct TranscriptionRequest {
    attachment_id:   i64,
    audio_file_path: String,
    language:        Option<String>,
    import_batch_id: Option<i64>,
    retry_count:     i32,
  }

src/model/result.rs
  struct TranscriptionResult {
    attachment_id:      i64,
    texto:              String,
    duration:           f64,
    model:              String,
    success:            bool,
    import_batch_id:    Option<i64>,
    error_message:      Option<String>,  // skip_serializing_if = None
    processing_time_ms: Option<u64>,     // skip_serializing_if = None; only on success
  }
  TranscriptionResult::success(attachment_id, texto, duration, model, import_batch_id, processing_time_ms) → Self
  TranscriptionResult::failure(attachment_id, model, import_batch_id, error_message)                       → Self

src/audio/decoder.rs
  fn decode(path: &Path) → Result<DecodedAudio, DecodeError>
  fn is_ogg_opus(path) → bool   // reads first 128 bytes for "OpusHead" magic
  OGG/Opus path → decode_ogg_opus() via libopus
  All other formats → Symphonia probe/decode loop

src/audio/resampler.rs
  fn to_mono_16k(decoded: &DecodedAudio) → Vec<f32>   // rubato SincFixedIn

src/audio/pipeline.rs
  struct AudioLimits { max_file_size_bytes: u64, max_duration_secs: f64 }
  struct ProcessedAudio { samples: Vec<f32>, duration_secs: f64 }
  fn process(path: &Path, limits: AudioLimits) → Result<ProcessedAudio, AudioError>
  Validation order: exists+size → extension → decode → duration → resample
  enum AudioError {
    FileNotFound(String),            // Deterministic
    UnsupportedFormat(String),       // Deterministic
    FileTooLarge { size_bytes, limit_bytes },    // Deterministic
    DurationExceeded { duration_secs, limit_secs }, // Deterministic
    Decode(DecodeError),             // Transient
    Resample(ResampleError),         // Transient
  }

src/whisper/model.rs
  struct WhisperModel { context: WhisperContext, name: String }
  unsafe impl Send + Sync   // required; context is not Send by default
  WhisperModel::load(path: &Path, name: String) → Result<Self, ModelError>

src/whisper/engine.rs
  #[derive(Clone)]
  struct WhisperEngine { model: Arc<WhisperModel> }
  WhisperEngine::new(model: Arc<WhisperModel>) → Self
  WhisperEngine::model_name() → &str
  WhisperEngine::transcribe(&self, samples: &[f32], language: Option<&str>)
    → Result<TranscriptionOutput, EngineError>
  Params: BeamSearch { beam_size: 5 }, no_speech_thold = 0.6
  WhisperState: allocated fresh per call (not reused across calls)
  struct TranscriptionOutput { text: String }
  enum EngineError { StateCreation(String), Inference(String), SegmentRead(String) }

src/messaging/rabbit.rs
  pub type Pool = deadpool_lapin::Pool
  fn build_pool(url: &str, max_size: usize) → Result<Pool, RabbitError>
  // Exchange / queue / routing-key string constants:
  WHISPER_EXCHANGE       = "whisper_exchange"
  WHISPER_QUEUE          = "whisper_transcriptions"
  WHISPER_ROUTING_KEY    = "transcription.request"
  RESULTS_EXCHANGE       = "whisper_results_exchange"
  RESULTS_QUEUE          = "whisper_results"
  RESULTS_ROUTING_KEY    = "transcription.result"
  RETRY_EXCHANGE         = "whisper_retry_exchange"
  RETRY_QUEUE            = "whisper_retry_queue"
  RETRY_ROUTING_KEY      = "transcription.retry"
  RETRY_TTL_MS           = 5000

src/messaging/consumer.rs
  struct Job { request: TranscriptionRequest, delivery: Delivery }
  struct RabbitConsumer
  RabbitConsumer::new(pool: &Pool, prefetch: u16) → Result<Self, ConsumerError>
  RabbitConsumer::into_receiver(self) → Result<mpsc::Receiver<Job>, ConsumerError>
  // Invalid JSON → NACK(requeue=false); valid → forwarded to channel

src/messaging/producer.rs
  #[derive(Clone)]
  struct RabbitProducer { pool: Pool, model_name: String }
  RabbitProducer::new(pool: &Pool, model_name: String) → Result<Self, ProducerError>
  RabbitProducer::publish_success(attachment_id, import_batch_id, texto, duration, processing_time_ms) → Result<(), ProducerError>
  RabbitProducer::publish_error(result: TranscriptionResult)   → Result<(), ProducerError>
  RabbitProducer::publish_retry(request: TranscriptionRequest) → Result<(), ProducerError>
  // publish_retry: increments retry_count; sets x-retry-count AMQP header

src/worker/task.rs
  async fn process(worker_id: usize, job: Job, engine: WhisperEngine,
                   producer: RabbitProducer, limits: AudioLimits,
                   metrics: Arc<Metrics>)
  enum TaskError { Deterministic(String), Transient(String) }
  // Deterministic → publish_error + ACK (no retry)
  // Transient     → handle_failure (retry via publish_retry, or final publish_error + ACK)
  // JoinError (panic in spawn_blocking) → treated as Transient
  // Any publish failure → NACK(requeue=true)
  // Blocking work dispatched via tokio::task::spawn_blocking:
  //   audio::pipeline::process → WhisperEngine::transcribe
  // Timing: Instant::now() captured immediately before spawn_blocking;
  //   elapsed().as_millis() read on success → stored as processing_time_ms in result

src/worker/pool.rs
  struct WorkerPool { workers_count: usize, engine: WhisperEngine,
                      producer: RabbitProducer, limits: AudioLimits,
                      metrics: Arc<Metrics> }
  WorkerPool::run(self, jobs_rx: mpsc::Receiver<Job>, shutdown_signal: ShutdownSignal)
  // N worker tasks share Arc<Mutex<Receiver>>
  // Channel capacity = workers_count × 2
```

---

### AMQP Topology

```
INPUT
  exchange     "whisper_exchange"           type=direct, durable
  queue        "whisper_transcriptions"     durable
  routing_key  "transcription.request"

OUTPUT
  exchange     "whisper_results_exchange"   type=direct, durable
  queue        "whisper_results"            durable
  routing_key  "transcription.result"

RETRY (Dead-Letter Exchange pattern)
  exchange     "whisper_retry_exchange"     type=direct, durable
  queue        "whisper_retry_queue"        durable, x-message-ttl=5000ms
  routing_key  "transcription.retry"
  DLX on TTL expiry → "whisper_exchange"   (message re-enters main queue)

  MAX_RETRIES = 2  →  3 total processing attempts
```

---

### Message Contracts

> Field names are domain-specific to the original application. Adapt for other use cases — the service only cares that the JSON is valid against the structs in `src/model/`.

**Request (inbound)**

```json
{
  "attachment_id":   123,
  "audio_file_path": "/mnt/audio/voice_001.ogg",
  "language":        "es",
  "import_batch_id": 42,
  "retry_count":     0
}
```

| Field | Rust type | Required | Notes |
|---|---|---|---|
| `attachment_id` | `i64` | ✅ | Asset identifier, echoed in result |
| `audio_file_path` | `String` | ✅ | Absolute path inside the container |
| `language` | `Option<String>` | ✗ | ISO 639-1 (`"es"`, `"en"`, …); `null` = auto-detect |
| `import_batch_id` | `Option<i64>` | ✗ | Passed through unchanged |
| `retry_count` | `i32` | ✗ | Managed by the service; send `0` or omit |

**Result (outbound — success)**

```json
{
  "attachment_id":      123,
  "texto":              "Hola, ¿cómo estás?",
  "duration":           4.32,
  "model":              "base-q5_1",
  "success":            true,
  "import_batch_id":    42,
  "processing_time_ms": 3241
}
```

**Result (outbound — failure)**

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

| Field | Rust type | Notes |
|---|---|---|
| `attachment_id` | `i64` | Echoed from request |
| `texto` | `String` | Full transcript; `""` on failure |
| `duration` | `f64` | Audio duration in seconds; `0.0` on failure |
| `model` | `String` | Model identifier used |
| `success` | `bool` | |
| `import_batch_id` | `Option<i64>` | Echoed from request |
| `error_message` | `Option<String>` | Serialized only when `success = false` (`skip_serializing_if`) |
| `processing_time_ms` | `Option<u64>` | Wall-clock ms from `spawn_blocking` start to its completion (decode + resample + inference). Serialized only when `success = true`. Absent on failure. |

---

### Error Classification

| `AudioError` variant | Class | Retry? |
|---|---|---|
| `FileNotFound` | Deterministic | ❌ |
| `UnsupportedFormat` | Deterministic | ❌ |
| `FileTooLarge` | Deterministic | ❌ |
| `DurationExceeded` | Deterministic | ❌ |
| `Decode` | Transient | ✅ |
| `Resample` | Transient | ✅ |
| `EngineError` (all variants) | Transient | ✅ |
| `JoinError` (spawn_blocking panic) | Transient | ✅ |

---

### Key Invariants

1. `AUDIO_SAMPLE_RATE` **must equal `16000`** — whisper.cpp hard requirement; changing this breaks inference.
2. `WhisperModel` is loaded **exactly once** at startup (expensive blocking call). `Arc<WhisperModel>` is cloned cheaply into each `WhisperEngine`.
3. `WhisperState` is **allocated fresh per `transcribe()` call** — never reused. This is what allows concurrent inference on the same model.
4. OGG/Opus detection reads the **first 128 bytes for the `"OpusHead"` magic string** — this check runs before the file extension check to correctly handle `.ogg` containers that carry Opus streams.
5. **Deterministic `AudioError` variants must never be retried** — they indicate a permanent condition (wrong format, missing file). Code in `task.rs` maps these to `TaskError::Deterministic`, which bypasses `handle_failure`.
6. `RABBITMQ_URL` default vhost **must use `%2F` encoding**, not a bare `/` — lapin will reject an unencoded slash.
7. Model file path is resolved as **`{MODELS_DIR}/ggml-{WHISPER_MODEL}.bin`** — both variables are required for the path to be valid.
8. `entrypoint.sh` **auto-downloads the model from Hugging Face** at container start if the file is missing.

---

### Docker / Build Notes

```
Builder stage:  rust:1-bookworm  +  libopus-dev  +  libclang-dev  +  cmake
Runtime stage:  debian:bookworm-slim  +  libopus0  +  libgomp1  +  libstdc++6

Dep-cache trick:
  1. COPY Cargo.toml + stub main.rs → cargo build --release  (caches all deps)
  2. COPY src/ → cargo build --release  (only recompiles ruistper itself)

After modifying src/:
  docker-compose build --no-cache ruistper
  (required to bypass the COPY src/ layer cache)
```

---

## 👥 Human Documentation

---

### What Is This?

Ruistper is an async Rust microservice that listens to a RabbitMQ queue, transcribes each audio file using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (GGML models), and publishes the result back to RabbitMQ. It is a drop-in replacement for the original Go + Python stack — the message format is identical.

> **Note on field names:** `attachment_id` and `import_batch_id` come from the original application's domain. The messaging contracts **can and should** be adapted to each use case — field names, types, and semantics are freely modifiable as long as both the producer and consumer agree.

---

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          Ruistper                            │
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

| Module | What it does |
|---|---|
| `main.rs` | Entry point, initialises tracing |
| `app.rs` | Wires everything together; handles startup and graceful shutdown |
| `config.rs` | Reads and validates all configuration from environment variables |
| `retry.rs` | Retry policy: `MAX_RETRIES = 2`, 5 s delay via Dead-Letter Exchange |
| `metrics.rs` | Atomic counters: received / succeeded / failed / retried / in-flight |
| `shutdown.rs` | Watch-channel pair + SIGINT/SIGTERM handler |
| `model/` | `TranscriptionRequest` and `TranscriptionResult` data structs |
| `audio/` | Full pipeline: validate → decode → resample to mono 16 kHz |
| `whisper/` | `WhisperModel` (loaded once at startup) + `WhisperEngine` (transcribes) |
| `messaging/` | AMQP connection pool, consumer, and producer |
| `worker/` | `WorkerPool` dispatcher + per-job `task::process` |

---

### Messaging Contracts

#### Input — Transcription Request

The service reads from:
- **Exchange:** `whisper_exchange` (direct)
- **Queue:** `whisper_transcriptions`
- **Routing key:** `transcription.request`

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
| `audio_file_path` | `String` | ✅ | Absolute path to the audio file, visible inside the container |
| `language` | `String` | ✗ | ISO 639-1 hint (`"es"`, `"en"`, …). `null` = auto-detect |
| `import_batch_id` | `i64` | ✗ | Passed through unchanged to the result |
| `retry_count` | `i32` | ✗ | Managed automatically on retry; send `0` or omit |

#### Output — Transcription Result

Published to:
- **Exchange:** `whisper_results_exchange` (direct)
- **Queue:** `whisper_results`
- **Routing key:** `transcription.result`

**Success:**
```json
{
  "attachment_id":      123,
  "texto":              "Hola, ¿cómo estás?",
  "duration":           4.32,
  "model":              "base-q5_1",
  "success":            true,
  "import_batch_id":    42,
  "processing_time_ms": 3241
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
| `processing_time_ms` | `i64 \| null` | Wall-clock processing time in ms (decode + resample + inference). Present only on success. Useful to compute RTF: `processing_time_ms / (duration × 1000)` |

---

### Retry & Error Handling

Jobs that fail due to a **transient error** (decode failure, inference error) are automatically retried up to 3 times total using a Dead-Letter Exchange pattern with a 5-second delay between attempts.

```
Main queue ──► [transient failure] ──► whisper_retry_exchange
                                             │
                                        whisper_retry_queue  (TTL = 5 s)
                                             │  x-dead-letter-exchange
                                             └──► whisper_exchange  (re-queued)
```

**Deterministic errors are never retried.** If the file is missing, the format is unsupported, or the file exceeds the size/duration limits, the service immediately publishes a failure result and acknowledges the message — no retry queue involved.

---

### Configuration

All settings come from environment variables. The service validates them at startup and will refuse to start if required values are missing or invalid.

| Variable | Default | Constraint | Description |
|---|---|---|---|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | Valid AMQP URL | Use `%2F` for the default vhost (not a bare `/`) |
| `RABBITMQ_QUEUE_NAME` | `whisper_transcriptions` | — | Input queue name |
| `WORKERS_COUNT` | `4` | ≥ 1 | Concurrent transcription workers |
| `WHISPER_MODEL` | `base` | — | Model ID, e.g. `base-q5_1`, `small`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` / `cuda` / `metal` | Inference device |
| `MODELS_DIR` | `/app/models` | — | Directory containing GGML `.bin` files |
| `MAX_FILE_SIZE_MB` | `100` | ≥ 1 | Maximum accepted file size |
| `MAX_AUDIO_DURATION_SEC` | `3600` | > 0 | Maximum accepted audio duration |
| `AUDIO_SAMPLE_RATE` | `16000` | **must be 16000** | whisper.cpp requires exactly 16 kHz |
| `TMP_DIR` | `/tmp/whisper` | — | Directory for intermediate files |
| `API_HOST` | `0.0.0.0` | — | HTTP API bind address (reserved, not yet active) |
| `API_PORT` | `8080` | 1–65535 | HTTP API port (reserved, not yet active) |
| `RUST_LOG` | `ruistper=info,warn` | tracing filter | Log level |

The model file is resolved as: `{MODELS_DIR}/ggml-{WHISPER_MODEL}.bin`

---

### Supported Audio Formats

| Format | Extension | Decoder |
|---|---|---|
| OGG / Opus | `.opus`, `.ogg` (OpusHead magic bytes) | `ogg` + `opus` crates via `libopus` |
| MP3 | `.mp3` | Symphonia |
| WAV | `.wav` | Symphonia |
| FLAC | `.flac` | Symphonia |
| AAC / M4A | `.aac`, `.m4a` | Symphonia |
| OGG / Vorbis | `.ogg` (non-Opus) | Symphonia |

All formats are resampled to **mono 16 kHz f32** before inference. `.wma` is intentionally not supported.

---

### Running with Docker Compose

```bash
# First run — builds the image, downloads the model, starts services
docker-compose up --build

# Rebuild after source changes (force-bypass layer cache)
docker-compose build --no-cache ruistper
docker-compose up
```

`entrypoint.sh` automatically downloads the GGML model from Hugging Face if it is missing:

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

#### Docker Compose example

Place this `docker-compose.yml` at the project root. Assumes `Ruistper/` is a sibling directory and models are stored in a `models/` bind mount.

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

  ruistper:
    build:
      context: ./Ruistper
      dockerfile: Dockerfile
    container_name: ruistper
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
      RUST_LOG: ruistper=info
    restart: unless-stopped
```

> **Audio file visibility:** Files must be accessible **inside the container**. Mount the audio directory as an additional volume (e.g. `- /data/audio:/mnt/audio`) and use the container-side path in `audio_file_path` (e.g. `/mnt/audio/voice_001.ogg`).

---

### Building Locally

**Prerequisites:**
- Rust 1.78+
- `libopus-dev` (for OGG/Opus decoding)
- `libclang-dev`, `cmake` (for whisper.cpp compilation)

```bash
cd Ruistper
cargo build --release
# binary: ./target/release/ruistper
```

---

### Dependencies

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
