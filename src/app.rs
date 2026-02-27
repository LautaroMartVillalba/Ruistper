use std::sync::Arc;

use crate::audio::AudioLimits;
use crate::config::Config;
use crate::messaging::{build_pool, RabbitConsumer, RabbitProducer};
use crate::metrics::Metrics;
use crate::shutdown;
use crate::whisper::{WhisperEngine, WhisperModel};
use crate::worker::WorkerPool;

// ── Error type ─────────────────────────────────────────────────────────────────

/// Top-level application error, surfaced only at startup.
/// Each variant wraps the underlying cause as a displayable string so
/// `main.rs` can log it cleanly without depending on every sub-module type.
#[derive(Debug)]
pub enum AppError {
    Config(crate::config::ConfigError),
    Io(std::io::Error),
    Model(crate::whisper::ModelError),
    RabbitMQ(crate::messaging::RabbitError),
    Consumer(crate::messaging::ConsumerError),
    Producer(crate::messaging::ProducerError),
    Pool(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(e)   => write!(f, "config error: {e}"),
            Self::Io(e)       => write!(f, "io error: {e}"),
            Self::Model(e)    => write!(f, "model load error: {e}"),
            Self::RabbitMQ(e) => write!(f, "rabbitmq pool error: {e}"),
            Self::Consumer(e) => write!(f, "consumer error: {e}"),
            Self::Producer(e) => write!(f, "producer error: {e}"),
            Self::Pool(e)     => write!(f, "worker pool error: {e}"),
        }
    }
}

// ── Entry point ────────────────────────────────────────────────────────────────

/// Full application lifecycle.
///
/// # Startup sequence
/// 1. Load and validate configuration from environment variables.
/// 2. Create the temporary audio directory (`TMP_DIR`).
/// 3. Build the RabbitMQ connection pool.
/// 4. Load the GGML Whisper model into memory (blocking; done on current thread
///    before the async plumbing is wired up, so there is no need for
///    `spawn_blocking` here — `#[tokio::main]` is already running but we call
///    this early before spawning tasks).
/// 5. Wire up consumer → producer → worker pool.
/// 6. Run until SIGINT / SIGTERM, then drain and exit.
pub async fn run() -> Result<(), AppError> {
    // ── 1. Configuration ──────────────────────────────────────────────────────
    let cfg = Config::load().map_err(AppError::Config)?;
    cfg.log_summary();

    // ── 2. Temporary directory ────────────────────────────────────────────────
    std::fs::create_dir_all(&cfg.tmp_dir).map_err(AppError::Io)?;
    tracing::debug!(path = %cfg.tmp_dir.display(), "ensured tmp_dir exists");

    // ── 3. Metrics ────────────────────────────────────────────────────────────
    let metrics = Arc::new(Metrics::new());

    // ── 4. Shutdown pair ──────────────────────────────────────────────────────
    // The handle is held here; the signal is cloned into the worker pool so it
    // can break its dispatch loop on demand.
    let (shutdown_handle, shutdown_signal) = shutdown::new_pair();

    // ── 5. RabbitMQ pool ──────────────────────────────────────────────────────
    // Allocate slightly more connections than workers so the consumer and
    // producer have dedicated channels without competing with workers.
    let pool_size = cfg.workers_count + 2;
    tracing::info!("🔌 connecting to RabbitMQ (pool_size={pool_size})...");
    let rabbit_pool = build_pool(&cfg.rabbitmq_url, pool_size)
        .await
        .map_err(AppError::RabbitMQ)?;
    tracing::info!("🔌 RabbitMQ connected");

    // ── 6. Whisper model ──────────────────────────────────────────────────────
    // `WhisperModel::load` is a blocking C-library call. We call it on the
    // current thread here (before any worker tasks are spawned) rather than
    // inside spawn_blocking — the tokio runtime is running but idle at this
    // point, so there is no risk of blocking an executor thread.
    let model_path = cfg.model_path();
    tracing::info!(
        model = %cfg.whisper_model,
        path  = %model_path.display(),
        "🤖 loading GGML model...",
    );
    let model = WhisperModel::load(&model_path, cfg.whisper_model.clone())
        .map_err(AppError::Model)?;
    tracing::info!(model = %cfg.whisper_model, "🤖 model loaded");

    let engine = WhisperEngine::new(Arc::new(model));

    // ── 7. Producer ───────────────────────────────────────────────────────────
    let producer = RabbitProducer::new(&rabbit_pool, cfg.whisper_model.clone())
        .await
        .map_err(AppError::Producer)?;

    // ── 8. Consumer ───────────────────────────────────────────────────────────
    // prefetch_count = workers_count mirrors Go's QoS setting.
    let consumer = RabbitConsumer::new(&rabbit_pool, cfg.workers_count as u16)
        .await
        .map_err(AppError::Consumer)?;

    // Spawns an internal consume_loop task and returns the job receiver.
    let jobs_rx = consumer
        .into_receiver()
        .await
        .map_err(AppError::Consumer)?;

    // ── 9. Audio limits ───────────────────────────────────────────────────────
    let limits = AudioLimits {
        max_file_size_bytes: cfg.max_file_size_bytes(),
        max_duration_secs:   cfg.max_audio_duration_sec,
    };

    // ── 10. Worker pool ───────────────────────────────────────────────────────
    let pool = WorkerPool::new(engine, producer, limits, cfg.workers_count, Arc::clone(&metrics));

    tracing::info!(
        workers = cfg.workers_count,
        "✅ Ruistper ready — waiting for transcription jobs"
    );

    // ── 11. Concurrent run + OS-signal wait ───────────────────────────────────
    // The pool runs in a background task so we can simultaneously wait for an
    // OS signal on the current task without blocking the pool dispatch loop.
    let pool_task = tokio::spawn(pool.run(jobs_rx, shutdown_signal));

    // Block until SIGINT or SIGTERM is received.
    shutdown::wait_for_os_signal().await;
    tracing::info!("🛑 signal received — initiating graceful shutdown...");

    // ── 12. Graceful shutdown ─────────────────────────────────────────────────
    // Trigger causes the pool dispatch loop to break after the current batch,
    // then it drops the internal channel so workers drain and exit.
    shutdown_handle.trigger();

    // Await the pool task; it returns only after all worker handles are joined.
    pool_task
        .await
        .map_err(|e| AppError::Pool(e.to_string()))?;

    metrics.log_summary();
    tracing::info!("✅ shutdown complete — goodbye");
    Ok(())
}
