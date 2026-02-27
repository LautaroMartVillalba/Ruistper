use std::path::Path;
use std::sync::Arc;

use lapin::{
    message::Delivery,
    options::{BasicAckOptions, BasicNackOptions},
};

use crate::audio::pipeline::{self, AudioError, AudioLimits};
use crate::messaging::{Job, RabbitProducer};
use crate::metrics::Metrics;
use crate::model::TranscriptionRequest;
use crate::retry::{RetryPolicy, MAX_RETRIES};
use crate::whisper::{EngineError, WhisperEngine};

// ── Internal error classification ─────────────────────────────────────────────

/// Classifies a processing error as deterministic or transient.
///
/// This drives the retry decision:
/// - [`Deterministic`] errors will not be resolved by retrying the same job.
///   They mirror Go's pre-validation checks (`FileExists`, `ValidateAudioExtension`)
///   which publish an error result and ACK immediately, bypassing the retry system.
/// - [`Transient`] errors may be resolved on retry. They go through
///   `handle_failure`, which mirrors Go's `handleFailure`.
#[derive(Debug)]
enum TaskError {
    Deterministic(String),
    Transient(String),
}

impl From<AudioError> for TaskError {
    fn from(e: AudioError) -> Self {
        match e {
            // Deterministic: file missing, wrong format, too large, too long.
            // These correspond exactly to Go's two pre-validation steps plus
            // the Python AudioProcessor.validate_file() checks.
            AudioError::FileNotFound(_)
            | AudioError::UnsupportedFormat(_)
            | AudioError::FileTooLarge { .. }
            | AudioError::DurationExceeded { .. } => Self::Deterministic(e.to_string()),

            // Transient: decode or resample errors may succeed on retry.
            AudioError::Decode(_) | AudioError::Resample(_) => Self::Transient(e.to_string()),
        }
    }
}

impl From<EngineError> for TaskError {
    fn from(e: EngineError) -> Self {
        // All whisper.cpp engine errors mirror Go's Python execution failures,
        // which are handled by handleFailure (retry-eligible).
        Self::Transient(e.to_string())
    }
}

// ── Public entry point ─────────────────────────────────────────────────────────

/// Process one transcription job end-to-end.
///
/// This is the Rust equivalent of Go's `processJob` in `worker/pool.go`,
/// with the Python processing pipeline (audio + whisper) merged into a single
/// `tokio::task::spawn_blocking` call.
///
/// # Execution flow
/// 1. Extract scalar fields from the job (avoid cloning the full request where possible).
/// 2. Dispatch blocking work to `spawn_blocking` (the **tokio bridge**):
///    - [`crate::audio::pipeline::process`] — validate, decode, resample to mono 16 kHz.
///    - [`crate::whisper::WhisperEngine::transcribe`] — run whisper.cpp inference.
/// 3. Route the outcome — mirrors Go's `processJob` + `handleFailure` logic exactly:
///    - `Ok((duration, text))` → `publish_success` + ACK.
///    - `Err(Deterministic)` → `publish_error` + ACK, **no retry**.
///    - `Err(Transient)` → `handle_failure` (retry or final error + ACK).
///    - `JoinError` (panic in blocking thread) → treated as transient.
///    - Any publish failure → NACK with `requeue = true`.
pub async fn process(
    worker_id: usize,
    job: Job,
    engine: WhisperEngine,
    producer: RabbitProducer,
    limits: AudioLimits,
    metrics: Arc<Metrics>,
) {
    // Destructure upfront so `delivery` stays accessible for ACK/NACK
    // after `request` fields are (cheaply) extracted for the closure.
    let Job { request, delivery } = job;

    let attachment_id = request.attachment_id;
    let import_batch_id = request.import_batch_id;
    let retry_count = request.retry_count;

    metrics.inc_in_flight();

    let retry_info = if retry_count > 0 {
        format!(" [retry {}/{}]", retry_count, MAX_RETRIES)
    } else {
        String::new()
    };

    tracing::info!(
        worker = worker_id,
        attachment_id,
        "▶️  job #{}{}",
        attachment_id,
        retry_info
    );

    // Clone only what the blocking closure needs; keep `request` alive
    // outside the closure so `handle_failure` can pass it to `publish_retry`.
    let path_str = request.audio_file_path.clone();
    let language = request.language.clone();

    // ── Tokio bridge ──────────────────────────────────────────────────────────
    // `audio::pipeline::process` (pydub equivalent) and `WhisperEngine::transcribe`
    // are synchronous and CPU-bound.  Spawning them on the blocking thread pool
    // keeps the async executor free for I/O (RabbitMQ, etc.).
    let blocking_result = tokio::task::spawn_blocking(move || {
        let path = Path::new(&path_str);

        // Step 1 — validate, decode, resample (equivalent to Python's AudioProcessor)
        let processed = pipeline::process(path, &limits).map_err(TaskError::from)?;

        // Step 2 — whisper.cpp inference (equivalent to Python's WhisperService)
        let output = engine
            .transcribe(&processed.samples, language.as_deref())
            .map_err(TaskError::from)?;

        Ok::<_, TaskError>((processed.duration_secs, output.text))
    })
    .await;

    match blocking_result {
        // ── spawn_blocking panicked ────────────────────────────────────────────
        // Treat as transient — mirrors Go's unhandled error path that calls handleFailure.
        Err(join_err) => {
            let msg = format!("blocking task panicked: {join_err}");
            tracing::error!(worker = worker_id, attachment_id, "{}", msg);
            handle_failure(
                worker_id,
                &delivery,
                &request,
                attachment_id,
                import_batch_id,
                retry_count,
                &producer,
                &metrics,
                msg,
            )
            .await;
        }

        // ── Deterministic failure ──────────────────────────────────────────────
        // Mirrors Go's FileExists / ValidateAudioExtension blocks:
        // publish_error + ACK immediately, never enter the retry system.
        Ok(Err(TaskError::Deterministic(msg))) => {
            metrics.inc_failed();
            tracing::warn!(
                worker = worker_id,
                attachment_id,
                "validation error (no retry): {}",
                msg
            );
            publish_error_and_ack(
                worker_id,
                &delivery,
                attachment_id,
                import_batch_id,
                &producer,
                msg,
            )
            .await;
        }

        // ── Transient failure ──────────────────────────────────────────────────
        // Mirrors Go's handleFailure call on Python execution errors.
        Ok(Err(TaskError::Transient(msg))) => {
            tracing::warn!(
                worker = worker_id,
                attachment_id,
                "processing error: {}",
                msg
            );
            handle_failure(
                worker_id,
                &delivery,
                &request,
                attachment_id,
                import_batch_id,
                retry_count,
                &producer,
                &metrics,
                msg,
            )
            .await;
        }

        // ── Success ────────────────────────────────────────────────────────────
        // Mirrors Go's PublishSuccess + Ack block (step 6 of processJob).
        Ok(Ok((duration, texto))) => {
            match producer
                .publish_success(attachment_id, import_batch_id, texto, duration)
                .await
            {
                Ok(_) => {
                    metrics.inc_succeeded();
                    tracing::info!(
                        worker = worker_id,
                        attachment_id,
                        "✅ #{} done ({:.1}s)",
                        attachment_id,
                        duration
                    );
                    let _ = delivery.ack(BasicAckOptions::default()).await;
                }
                Err(e) => {
                    // Mirrors Go: log + Nack(false, true)
                    tracing::error!(
                        worker = worker_id,
                        attachment_id,
                        error = %e,
                        "❌ publish failed, NACKing"
                    );
                    let _ = delivery
                        .nack(BasicNackOptions {
                            multiple: false,
                            requeue: true,
                        })
                        .await;
                }
            }
        }
    }
    metrics.dec_in_flight();
}

// ── Private helpers ────────────────────────────────────────────────────────────

/// Retry or give up, mirroring Go's `handleFailure` in `worker/pool.go` exactly.
///
/// ```text
/// Go:  if rabbitmq.ShouldRetry(request.RetryCount) { PublishRetry; Ack }
///      else { PublishError; Ack }
///      // on any publish failure → Nack(false, true)
/// ```
async fn handle_failure(
    worker_id: usize,
    delivery: &Delivery,
    request: &TranscriptionRequest,
    attachment_id: i64,
    import_batch_id: Option<i64>,
    retry_count: i32,
    producer: &RabbitProducer,
    metrics: &Metrics,
    error_message: String,
) {
    let policy = RetryPolicy::default();

    if policy.should_retry(retry_count) {
        metrics.inc_retried();
        // Mirrors Go: log + PublishRetry + Ack
        tracing::info!(
            worker = worker_id,
            attachment_id,
            "🔄 #{} retry {}/{}",
            attachment_id,
            retry_count + 1,
            MAX_RETRIES
        );

        match producer.publish_retry(request).await {
            Ok(_) => {
                let _ = delivery.ack(BasicAckOptions::default()).await;
            }
            Err(e) => {
                tracing::error!(
                    worker = worker_id,
                    attachment_id,
                    error = %e,
                    "❌ retry publish failed, NACKing"
                );
                let _ = delivery
                    .nack(BasicNackOptions {
                        multiple: false,
                        requeue: true,
                    })
                    .await;
            }
        }
    } else {
        metrics.inc_failed();
        // Max retries exceeded — mirrors Go: log + PublishError + Ack
        tracing::error!(
            worker = worker_id,
            attachment_id,
            "❌ #{} failed (max retries): {}",
            attachment_id,
            error_message
        );
        publish_error_and_ack(
            worker_id,
            delivery,
            attachment_id,
            import_batch_id,
            producer,
            error_message,
        )
        .await;
    }
}

/// Publish a final error result and ACK.
/// On publish failure, NACK with requeue=true — mirrors all Go error-publish blocks.
async fn publish_error_and_ack(
    worker_id: usize,
    delivery: &Delivery,
    attachment_id: i64,
    import_batch_id: Option<i64>,
    producer: &RabbitProducer,
    error_message: String,
) {
    match producer
        .publish_error(attachment_id, import_batch_id, error_message)
        .await
    {
        Ok(_) => {
            let _ = delivery.ack(BasicAckOptions::default()).await;
        }
        Err(e) => {
            // Mirrors Go: log + Nack(false, true)
            tracing::error!(
                worker = worker_id,
                attachment_id,
                error = %e,
                "❌ error publish failed, NACKing"
            );
            let _ = delivery
                .nack(BasicNackOptions {
                    multiple: false,
                    requeue: true,
                })
                .await;
        }
    }
}
