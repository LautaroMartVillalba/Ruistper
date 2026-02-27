use serde::{Deserialize, Serialize};

/// Transcription result published back to RabbitMQ.
///
/// Published to: `whisper_results_exchange` (direct)
/// Routing key:  `transcription.result`
/// Queue:        `whisper_results`
///
/// The service always publishes exactly one result per received job,
/// whether successful or failed.
///
/// Mirrors Go's `TranscriptionResult` in `internal/rabbitmq/types.go`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Same value received in the request.
    pub attachment_id: i64,

    /// Transcribed text. Empty string on failure.
    pub texto: String,

    /// Audio duration in seconds. `0.0` on failure.
    pub duration: f64,

    /// Whisper model name used (e.g. "base", "base.q5_0").
    pub model: String,

    /// `true` if transcription succeeded, `false` on any error.
    pub success: bool,

    /// Same value received in the request. Absent if not provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub import_batch_id: Option<i64>,

    /// Human-readable error description. Only present when `success` is `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

impl TranscriptionResult {
    /// Builds a successful result.
    pub fn success(
        attachment_id: i64,
        import_batch_id: Option<i64>,
        texto: String,
        duration: f64,
        model: String,
    ) -> Self {
        Self {
            attachment_id,
            texto,
            duration,
            model,
            success: true,
            import_batch_id,
            error_message: None,
        }
    }

    /// Builds a failed result.
    pub fn failure(
        attachment_id: i64,
        import_batch_id: Option<i64>,
        model: String,
        error_message: String,
    ) -> Self {
        Self {
            attachment_id,
            texto: String::new(),
            duration: 0.0,
            model,
            success: false,
            import_batch_id,
            error_message: Some(error_message),
        }
    }
}
