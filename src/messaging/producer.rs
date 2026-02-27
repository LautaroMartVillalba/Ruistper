use lapin::{
    options::{
        BasicPublishOptions, ExchangeDeclareOptions, QueueBindOptions, QueueDeclareOptions,
    },
    types::{AMQPValue, FieldTable},
    BasicProperties, Channel, ExchangeKind,
};

use crate::model::{TranscriptionRequest, TranscriptionResult};

use super::rabbit::{
    Pool, MAIN_EXCHANGE, MAIN_ROUTING_KEY, RESULTS_EXCHANGE, RESULTS_QUEUE,
    RESULTS_ROUTING_KEY, RETRY_EXCHANGE, RETRY_QUEUE, RETRY_ROUTING_KEY, RETRY_TTL_MS,
};

// ── Error ──────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ProducerError {
    Connection(String),
    Channel(String),
    Topology(String),
    Serialize(String),
    Publish(String),
}

impl std::fmt::Display for ProducerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connection(m) => write!(f, "producer connection error: {m}"),
            Self::Channel(m) => write!(f, "producer channel error: {m}"),
            Self::Topology(m) => write!(f, "topology declaration failed: {m}"),
            Self::Serialize(m) => write!(f, "serialization failed: {m}"),
            Self::Publish(m) => write!(f, "publish failed: {m}"),
        }
    }
}

impl std::error::Error for ProducerError {}

// ── RabbitProducer ─────────────────────────────────────────────────────────────

/// RabbitMQ producer.
///
/// Holds a single AMQP channel for all outbound publishing (results + retries).
/// The channel keeps the parent connection alive via `Arc` (lapin is Arc-backed).
///
/// # Sharing across workers
/// `RabbitProducer` implements `Clone` — cloning is cheap (Arc increment on the
/// channel). Each worker can hold its own clone and publish concurrently; lapin
/// serialises writes to the underlying channel internally.
/// Mirrors Go's single `*Producer` passed by pointer to all workers.
#[derive(Clone)]
pub struct RabbitProducer {
    /// AMQP channel. Also keeps the parent connection alive (lapin is Arc-backed).
    channel: Channel,
    /// Whisper model name embedded in every result message. Mirrors Go's `producer.model`.
    model_name: String,
}

impl RabbitProducer {
    /// Create the producer: obtain a connection from `pool`, open a channel,
    /// and declare the full producer-side topology (results + retry).
    pub async fn new(pool: &Pool, model_name: String) -> Result<Self, ProducerError> {
        let conn = pool
            .get()
            .await
            .map_err(|e| ProducerError::Connection(e.to_string()))?;

        let channel = conn
            .create_channel()
            .await
            .map_err(|e| ProducerError::Channel(e.to_string()))?;

        // conn (pool Object) drops here; channel's Arc<Connection> keeps the
        // underlying TCP connection alive.

        declare_topology(&channel).await?;

        tracing::info!("[Producer] connected and ready");

        Ok(Self { channel, model_name })
    }

    // ── Public publish API ─────────────────────────────────────────────────────
    // All three methods mirror Go's Producer API in rabbitmq/producer.go exactly.

    /// Publish a successful transcription result to `whisper_results`.
    ///
    /// Mirrors Go's `Producer.PublishSuccess`.
    pub async fn publish_success(
        &self,
        attachment_id: i64,
        import_batch_id: Option<i64>,
        texto: String,
        duration: f64,
    ) -> Result<(), ProducerError> {
        let result = TranscriptionResult::success(
            attachment_id,
            import_batch_id,
            texto,
            duration,
            self.model_name.clone(),
        );
        self.publish_result(&result).await
    }

    /// Publish a failed result to `whisper_results`.
    ///
    /// Mirrors Go's `Producer.PublishError`.
    pub async fn publish_error(
        &self,
        attachment_id: i64,
        import_batch_id: Option<i64>,
        error_message: String,
    ) -> Result<(), ProducerError> {
        let result = TranscriptionResult::failure(
            attachment_id,
            import_batch_id,
            self.model_name.clone(),
            error_message,
        );
        self.publish_result(&result).await
    }

    /// Increment `retry_count`, publish to `whisper_retry_queue`, and embed the
    /// updated count in both the message body and the AMQP header `x-retry-count`.
    ///
    /// The retry queue's TTL (5 s) and DLX route the message back to
    /// `whisper_transcriptions` automatically.
    ///
    /// Mirrors Go's `Producer.PublishRetry`, which increments `request.RetryCount`
    /// and sets `Headers["x-retry-count"] = int32(request.RetryCount)`.
    pub async fn publish_retry(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<(), ProducerError> {
        // Clone and increment — mirrors Go: `request.RetryCount++`
        let mut retried = request.clone();
        retried.retry_count += 1;

        let body = serde_json::to_vec(&retried)
            .map_err(|e| ProducerError::Serialize(e.to_string()))?;

        // Header x-retry-count as LongInt (i32) — mirrors Go's `int32(request.RetryCount)`
        let mut headers = FieldTable::default();
        headers.insert("x-retry-count".into(), AMQPValue::LongInt(retried.retry_count));

        let props = BasicProperties::default()
            .with_content_type("application/json".into())
            .with_delivery_mode(2) // persistent — mirrors Go's amqp.Persistent
            .with_headers(headers);

        self.channel
            .basic_publish(
                RETRY_EXCHANGE,
                RETRY_ROUTING_KEY,
                BasicPublishOptions::default(),
                &body,
                props,
            )
            .await
            .map_err(|e| ProducerError::Publish(e.to_string()))?;

        Ok(())
    }

    /// Return the model name stored in this producer.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    // ── Private helpers ────────────────────────────────────────────────────────
    /// Persistent delivery mode, `application/json` content type.
    /// Mirrors Go's `Producer.PublishResult`.
    async fn publish_result(&self, result: &TranscriptionResult) -> Result<(), ProducerError> {
        let body = serde_json::to_vec(result)
            .map_err(|e| ProducerError::Serialize(e.to_string()))?;

        let props = BasicProperties::default()
            .with_content_type("application/json".into())
            .with_delivery_mode(2); // persistent — mirrors Go's amqp.Persistent

        self.channel
            .basic_publish(
                RESULTS_EXCHANGE,
                RESULTS_ROUTING_KEY,
                BasicPublishOptions::default(),
                &body,
                props,
            )
            .await
            .map_err(|e| ProducerError::Publish(e.to_string()))?;

        Ok(())
    }
}

// ── Topology ───────────────────────────────────────────────────────────────────

// NOTE: `should_retry` and `MAX_RETRIES` live in `crate::retry`.
// The `producer` module only handles publishing; retry decisions belong to the worker.

/// Declare the producer-side AMQP topology (results + retry).
///
/// Mirrors Go's `declareProducerTopology` in `rabbitmq/producer.go`:
///
/// **Results:**
/// - Exchange `whisper_results_exchange` (direct, durable)
/// - Queue `whisper_results` (durable)
/// - Binding via `transcription.result`
///
/// **Retry:**
/// - Exchange `whisper_retry_exchange` (direct, durable)
/// - Queue `whisper_retry_queue` (durable, TTL=5000ms, DLX→`whisper_exchange`,
///   DLX routing key=`transcription.request`)
/// - Binding via `transcription.retry`
async fn declare_topology(channel: &Channel) -> Result<(), ProducerError> {
    // ── Results ──────────────────────────────────────────────────────────────

    channel
        .exchange_declare(
            RESULTS_EXCHANGE,
            ExchangeKind::Direct,
            ExchangeDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|e| ProducerError::Topology(format!("exchange '{RESULTS_EXCHANGE}': {e}")))?;

    channel
        .queue_declare(
            RESULTS_QUEUE,
            QueueDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|e| ProducerError::Topology(format!("queue '{RESULTS_QUEUE}': {e}")))?;

    channel
        .queue_bind(
            RESULTS_QUEUE,
            RESULTS_EXCHANGE,
            RESULTS_ROUTING_KEY,
            QueueBindOptions::default(),
            FieldTable::default(),
        )
        .await
        .map_err(|e| {
            ProducerError::Topology(format!(
                "bind '{RESULTS_QUEUE}' → '{RESULTS_EXCHANGE}': {e}"
            ))
        })?;

    // ── Retry ─────────────────────────────────────────────────────────────────

    channel
        .exchange_declare(
            RETRY_EXCHANGE,
            ExchangeKind::Direct,
            ExchangeDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|e| ProducerError::Topology(format!("exchange '{RETRY_EXCHANGE}': {e}")))?;

    // Retry queue — durable with TTL and Dead Letter Exchange back to the main queue.
    // Mirrors Go's QueueDeclare args:
    //   "x-message-ttl":             int32(5000)
    //   "x-dead-letter-exchange":    "whisper_exchange"
    //   "x-dead-letter-routing-key": "transcription.request"
    let mut retry_args = FieldTable::default();
    retry_args.insert("x-message-ttl".into(), AMQPValue::LongInt(RETRY_TTL_MS));
    retry_args.insert(
        "x-dead-letter-exchange".into(),
        AMQPValue::LongString(MAIN_EXCHANGE.as_bytes().to_vec().into()),
    );
    retry_args.insert(
        "x-dead-letter-routing-key".into(),
        AMQPValue::LongString(MAIN_ROUTING_KEY.as_bytes().to_vec().into()),
    );

    channel
        .queue_declare(
            RETRY_QUEUE,
            QueueDeclareOptions {
                durable: true,
                ..Default::default()
            },
            retry_args,
        )
        .await
        .map_err(|e| ProducerError::Topology(format!("queue '{RETRY_QUEUE}': {e}")))?;

    channel
        .queue_bind(
            RETRY_QUEUE,
            RETRY_EXCHANGE,
            RETRY_ROUTING_KEY,
            QueueBindOptions::default(),
            FieldTable::default(),
        )
        .await
        .map_err(|e| {
            ProducerError::Topology(format!(
                "bind '{RETRY_QUEUE}' → '{RETRY_EXCHANGE}': {e}"
            ))
        })?;

    Ok(())
}
