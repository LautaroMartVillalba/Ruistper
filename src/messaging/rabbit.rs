use std::time::Duration;

use deadpool_lapin::Manager;
use lapin::ConnectionProperties;

/// Re-exported so other modules inside `messaging/` can import Pool from here.
pub type Pool = deadpool_lapin::Pool;

// ── Consumer-side topology ─────────────────────────────────────────────────────
// Mirrors Go's constants in rabbitmq/consumer.go

/// Direct exchange from which the service reads transcription requests.
pub const MAIN_EXCHANGE: &str = "whisper_exchange";
/// Durable queue bound to [`MAIN_EXCHANGE`] for incoming jobs.
pub const MAIN_QUEUE: &str = "whisper_transcriptions";
/// Routing key used when publishing transcription requests into [`MAIN_EXCHANGE`].
pub const MAIN_ROUTING_KEY: &str = "transcription.request";

// ── Producer-side topology — results ──────────────────────────────────────────
// Mirrors Go's constants in rabbitmq/producer.go

/// Direct exchange to which completed results are published.
pub const RESULTS_EXCHANGE: &str = "whisper_results_exchange";
/// Durable queue that collects completed transcription results.
pub const RESULTS_QUEUE: &str = "whisper_results";
/// Routing key for result messages.
pub const RESULTS_ROUTING_KEY: &str = "transcription.result";

// ── Producer-side topology — retry ────────────────────────────────────────────

/// Direct exchange for retry messages.
pub const RETRY_EXCHANGE: &str = "whisper_retry_exchange";
/// Routing key for retry messages.
pub const RETRY_ROUTING_KEY: &str = "transcription.retry";
/// Durable queue with TTL and DLX that re-routes expired messages back to the main queue.
pub const RETRY_QUEUE: &str = "whisper_retry_queue";

/// `x-message-ttl` on the retry queue in milliseconds.
/// After this delay, RabbitMQ routes the message back to [`MAIN_EXCHANGE`] via the DLX.
/// Mirrors Go's `RetryTTLMs = 5000`.
pub const RETRY_TTL_MS: i32 = 5_000;

// NOTE: MAX_RETRIES lives in `crate::retry` — the single source of truth for
// the retry limit. `messaging` has no dependency on `retry`.

// ── Connection retry ───────────────────────────────────────────────────────────
// Mirrors Go's Connect() in rabbitmq/connection.go (maxRetries=10, retryInterval=5s).

const MAX_CONNECT_ATTEMPTS: u32 = 10;
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(5);

// ── Error ──────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum RabbitError {
    /// Could not establish a connection after all retry attempts.
    Connection(String),
    /// Failed to build the connection pool itself.
    Pool(String),
}

impl std::fmt::Display for RabbitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connection(msg) => write!(f, "RabbitMQ connection failed: {msg}"),
            Self::Pool(msg) => write!(f, "connection pool build failed: {msg}"),
        }
    }
}

impl std::error::Error for RabbitError {}

// ── Pool constructor ───────────────────────────────────────────────────────────

/// Build a [`deadpool_lapin`] connection pool and verify connectivity with retry logic.
///
/// Attempts up to [`MAX_CONNECT_ATTEMPTS`] (10) times with a
/// [`CONNECT_RETRY_INTERVAL`] (5 s) delay between each attempt, mirroring Go's
/// `Connect()` in `rabbitmq/connection.go`.
///
/// `max_connections` should be at least `WORKERS_COUNT + 2` to cover the
/// consumer and producer channels as well as any headroom.
pub async fn build_pool(url: &str, max_connections: usize) -> Result<Pool, RabbitError> {
    let manager = Manager::new(url, ConnectionProperties::default());

    let pool = Pool::builder(manager)
        .max_size(max_connections)
        .build()
        .map_err(|e| RabbitError::Pool(e.to_string()))?;

    for attempt in 1..=MAX_CONNECT_ATTEMPTS {
        match pool.get().await {
            Ok(_) => {
                tracing::info!("📡 RabbitMQ connected");
                return Ok(pool);
            }
            Err(e) if attempt < MAX_CONNECT_ATTEMPTS => {
                tracing::warn!(
                    attempt,
                    max = MAX_CONNECT_ATTEMPTS,
                    "⚠️  RabbitMQ not ready, retrying in {}s...",
                    CONNECT_RETRY_INTERVAL.as_secs()
                );
                tokio::time::sleep(CONNECT_RETRY_INTERVAL).await;
            }
            Err(e) => {
                return Err(RabbitError::Connection(format!(
                    "failed after {MAX_CONNECT_ATTEMPTS} attempts: {e}"
                )));
            }
        }
    }

    unreachable!()
}
