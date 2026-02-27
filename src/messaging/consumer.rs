use futures_util::StreamExt;
use lapin::{
    message::Delivery,
    options::{
        BasicConsumeOptions, BasicNackOptions, BasicQosOptions, ExchangeDeclareOptions,
        QueueBindOptions, QueueDeclareOptions,
    },
    types::FieldTable,
    Channel, Consumer as LapinConsumer, ExchangeKind,
};
use tokio::sync::mpsc;

use crate::model::TranscriptionRequest;

use super::rabbit::{
    Pool, MAIN_EXCHANGE, MAIN_QUEUE, MAIN_ROUTING_KEY,
};

// ── Public types ───────────────────────────────────────────────────────────────

/// A parsed transcription job ready for processing.
///
/// Carries both the deserialized request and the raw lapin [`Delivery`].
/// The worker is responsible for calling `delivery.ack()` / `delivery.nack()`.
/// Mirrors Go's `Job` in `rabbitmq/consumer.go`.
pub struct Job {
    pub request: TranscriptionRequest,
    /// Raw AMQP delivery — used by workers to ACK or NACK after processing.
    pub delivery: Delivery,
}

// ── Error ──────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ConsumerError {
    Connection(String),
    Channel(String),
    Topology(String),
    Qos(String),
    Start(String),
}

impl std::fmt::Display for ConsumerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connection(m) => write!(f, "consumer connection error: {m}"),
            Self::Channel(m) => write!(f, "consumer channel error: {m}"),
            Self::Topology(m) => write!(f, "topology declaration failed: {m}"),
            Self::Qos(m) => write!(f, "QoS setup failed: {m}"),
            Self::Start(m) => write!(f, "failed to start consuming: {m}"),
        }
    }
}

impl std::error::Error for ConsumerError {}

// ── RabbitConsumer ─────────────────────────────────────────────────────────────

/// RabbitMQ consumer.
///
/// Holds its own AMQP channel (which keeps the underlying connection alive via `Arc`).
/// Call [`into_receiver`](Self::into_receiver) to start consuming and obtain the
/// job channel used by the worker pool.
pub struct RabbitConsumer {
    /// AMQP channel. Also keeps the parent connection alive (lapin is Arc-backed).
    channel: Channel,
    prefetch_count: u16,
}

impl RabbitConsumer {
    /// Create the consumer: obtain a connection from `pool`, open a channel,
    /// declare the consumer-side topology, and configure QoS.
    ///
    /// `prefetch_count` is set equal to `WORKERS_COUNT`, mirroring Go's
    /// `channel.Qos(prefetchCount, 0, false)` — the broker pushes no more
    /// unacked messages than there are available workers.
    pub async fn new(pool: &Pool, prefetch_count: u16) -> Result<Self, ConsumerError> {
        let conn = pool
            .get()
            .await
            .map_err(|e| ConsumerError::Connection(e.to_string()))?;

        let channel = conn
            .create_channel()
            .await
            .map_err(|e| ConsumerError::Channel(e.to_string()))?;

        // conn (pool Object) drops here; channel's internal Arc<Connection> keeps
        // the underlying TCP connection alive for the lifetime of the channel.

        declare_topology(&channel).await?;

        channel
            .basic_qos(prefetch_count, BasicQosOptions { global: false })
            .await
            .map_err(|e| ConsumerError::Qos(e.to_string()))?;

        tracing::info!(queue = MAIN_QUEUE, prefetch = prefetch_count, "consumer ready");

        Ok(Self { channel, prefetch_count })
    }

    /// Start consuming and return the receiver end of the job channel.
    ///
    /// Spawns a background task that:
    /// 1. Reads deliveries from the lapin consumer stream.
    /// 2. NACKs with `requeue=false` on JSON parse errors (mirrors Go: `msg.Nack(false, false)`).
    /// 3. Overlays `retry_count` from the AMQP header `x-retry-count` (mirrors Go consumer.go).
    /// 4. Forwards valid [`Job`]s to the returned `mpsc::Receiver`.
    ///
    /// Channel capacity = `prefetch_count * 2`, matching Go's buffered channel pattern.
    /// Mirrors Go's `consumer.Consume()` which returns `<-chan Job`.
    pub async fn into_receiver(self) -> Result<mpsc::Receiver<Job>, ConsumerError> {
        let capacity = (self.prefetch_count as usize) * 2;
        let (tx, rx) = mpsc::channel::<Job>(capacity);

        // Register consumer on the channel. Consumer tag mirrors Go's "go-orchestrator".
        let lapin_consumer = self
            .channel
            .basic_consume(
                MAIN_QUEUE,
                "rust-orchestrator",
                BasicConsumeOptions {
                    no_ack: false, // manual ACK — mirrors Go's auto-ack: false
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await
            .map_err(|e| ConsumerError::Start(e.to_string()))?;

        // The lapin Consumer holds an Arc reference to the channel (and thus the
        // connection). Spawning it moves the channel's lifetime into the task.
        tokio::spawn(consume_loop(lapin_consumer, tx));

        tracing::info!(queue = MAIN_QUEUE, "▶️  consuming");

        Ok(rx)
    }
}

// ── Background task ────────────────────────────────────────────────────────────

/// Maps raw lapin deliveries into [`Job`] items.
/// Runs as a persistent `tokio::spawn`ed task for the application lifetime.
async fn consume_loop(mut consumer: LapinConsumer, tx: mpsc::Sender<Job>) {
    while let Some(result) = consumer.next().await {
        let delivery = match result {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(error = %e, "consumer stream error");
                break;
            }
        };

        let request = match parse_delivery(&delivery) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "⚠️  invalid message — NACKing without requeue");
                // Mirrors Go: msg.Nack(false, false)
                let _ = delivery
                    .nack(BasicNackOptions {
                        multiple: false,
                        requeue: false,
                    })
                    .await;
                continue;
            }
        };

        let job = Job { request, delivery };

        if tx.send(job).await.is_err() {
            // The receiver was dropped — orchestrator is shutting down.
            break;
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Deserialize the delivery body into a [`TranscriptionRequest`] and overlay
/// `retry_count` from the AMQP header `x-retry-count`.
///
/// Header extraction mirrors Go consumer.go:
/// ```go
/// if retryCount, ok := msg.Headers["x-retry-count"].(int32); ok { ... }
/// else if retryCount, ok := msg.Headers["x-retry-count"].(int64); ok { ... }
/// ```
fn parse_delivery(delivery: &Delivery) -> Result<TranscriptionRequest, String> {
    let mut request: TranscriptionRequest = serde_json::from_slice(&delivery.data)
        .map_err(|e| format!("JSON parse error: {e}"))?;

    if let Some(count) = extract_retry_count(delivery) {
        request.retry_count = count;
    }

    Ok(request)
}

/// Extract `x-retry-count` from AMQP headers, accepting LongInt (i32) or
/// LongLongInt (i64) — mirrors Go's dual type-assertion.
fn extract_retry_count(delivery: &Delivery) -> Option<i32> {
    use lapin::types::AMQPValue;

    delivery
        .properties
        .headers()
        .as_ref()?
        .inner()
        .get("x-retry-count")
        .and_then(|v| match v {
            AMQPValue::LongInt(n) => Some(*n),
            AMQPValue::LongLongInt(n) => Some(*n as i32),
            AMQPValue::ShortInt(n) => Some(*n as i32),
            AMQPValue::ShortShortInt(n) => Some(*n as i32),
            _ => None,
        })
}

/// Declare the consumer-side AMQP topology.
///
/// Mirrors Go's `declareConsumerTopology` in `rabbitmq/consumer.go`:
/// - Exchange `whisper_exchange` (direct, durable)
/// - Queue `whisper_transcriptions` (durable)
/// - Binding: queue ← exchange via `transcription.request`
async fn declare_topology(channel: &Channel) -> Result<(), ConsumerError> {
    // Main exchange — direct, durable
    channel
        .exchange_declare(
            MAIN_EXCHANGE,
            ExchangeKind::Direct,
            ExchangeDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|e| ConsumerError::Topology(format!("exchange '{MAIN_EXCHANGE}': {e}")))?;

    // Main queue — durable
    channel
        .queue_declare(
            MAIN_QUEUE,
            QueueDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|e| ConsumerError::Topology(format!("queue '{MAIN_QUEUE}': {e}")))?;

    // Bind queue to exchange
    channel
        .queue_bind(
            MAIN_QUEUE,
            MAIN_EXCHANGE,
            MAIN_ROUTING_KEY,
            QueueBindOptions::default(),
            FieldTable::default(),
        )
        .await
        .map_err(|e| {
            ConsumerError::Topology(format!(
                "bind '{MAIN_QUEUE}' → '{MAIN_EXCHANGE}' via '{MAIN_ROUTING_KEY}': {e}"
            ))
        })?;

    Ok(())
}
