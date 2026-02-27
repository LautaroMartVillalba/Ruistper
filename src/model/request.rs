use serde::{Deserialize, Serialize};

/// Incoming transcription job received from RabbitMQ.
///
/// Published to: `whisper_exchange` (direct)
/// Routing key:  `transcription.request`
/// Queue:        `whisper_transcriptions`
///
/// Mirrors Go's `TranscriptionRequest` in `internal/rabbitmq/types.go`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptionRequest {
    /// Unique job identifier. Returned unchanged in the result for correlation.
    pub attachment_id: i64,

    /// Absolute path to the audio file, accessible from the container filesystem.
    pub audio_file_path: String,

    /// ISO 639-1 language code (e.g. "es", "en"). If absent, Whisper auto-detects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Optional batch grouping identifier. Passed through without modification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub import_batch_id: Option<i64>,

    /// Number of times this job has already been attempted. Defaults to 0.
    /// Also carried in the AMQP header `x-retry-count`.
    #[serde(default)]
    pub retry_count: i32,
}
