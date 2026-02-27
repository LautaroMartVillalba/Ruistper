use std::sync::Arc;

use whisper_rs::{FullParams, SamplingStrategy};

use super::model::WhisperModel;

/// Output of a successful transcription.
pub struct TranscriptionOutput {
    /// Full transcript assembled from all whisper segments.
    pub text: String,
}

/// Errors that can occur during inference.
#[derive(Debug)]
pub enum EngineError {
    /// Failed to allocate a `WhisperState` from the context.
    StateCreation(String),
    /// `state.full()` returned an error.
    Inference(String),
    /// Reading a segment from the completed state failed.
    SegmentRead(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateCreation(msg) => write!(f, "whisper state creation failed: {msg}"),
            Self::Inference(msg) => write!(f, "whisper inference failed: {msg}"),
            Self::SegmentRead(msg) => write!(f, "segment read failed: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Wraps a shared [`WhisperModel`] and exposes a `transcribe()` method.
///
/// Holds the model via `Arc` — cloning the engine is cheap (just increments
/// the reference count) and does not reload the model.
///
/// # Concurrency
/// Each `transcribe()` call allocates a fresh `WhisperState` from the shared
/// context. States are independent, so multiple threads can call `transcribe()`
/// on the **same** `WhisperEngine` simultaneously without contention on the model
/// weights.
#[derive(Clone)]
pub struct WhisperEngine {
    model: Arc<WhisperModel>,
}

impl WhisperEngine {
    /// Create an engine from a loaded model.
    pub fn new(model: Arc<WhisperModel>) -> Self {
        Self { model }
    }

    /// Name of the underlying model (e.g. `"base.q5_0"`).
    /// Used by workers when building [`crate::model::TranscriptionResult`].
    pub fn model_name(&self) -> &str {
        &self.model.name
    }

    /// Transcribe mono 16 kHz f32 PCM samples to text.
    ///
    /// # Arguments
    /// * `samples`  — mono 16 kHz f32 PCM buffer, as produced by
    ///   [`crate::audio::pipeline::process`].
    /// * `language` — ISO 639-1 language code (`"es"`, `"en"`, …).
    ///   Pass `None` to let whisper.cpp auto-detect the language.
    ///
    /// # How it works
    /// 1. Allocates a `WhisperState` from the shared context (cheap — only
    ///    buffers, not model weights).
    /// 2. Configures `FullParams` to mirror the Python settings:
    ///    beam search with `beam_size=5`, silence threshold (`no_speech_thold=0.6`)
    ///    equivalent to `vad_filter=True` / `min_silence_duration_ms=500`.
    /// 3. Runs inference via `state.full()`.
    /// 4. Collects all segments, trims whitespace, and joins them with a single space.
    ///
    /// # Errors
    /// Returns [`EngineError`] on state allocation failure, inference failure,
    /// or segment read failure.
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionOutput, EngineError> {
        // Allocate per-call inference state. This does NOT reload model weights.
        let mut state = self
            .model
            .context
            .create_state()
            .map_err(|e| EngineError::StateCreation(e.to_string()))?;

        // Mirror Python faster-whisper settings:
        //   beam_size=5 → BeamSearch { beam_size: 5 }
        //   vad_filter + min_silence_duration_ms=500 → no_speech_thold=0.6
        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });

        params.set_language(language);

        // Silence-based VAD: segments below this probability are discarded.
        params.set_no_speech_thold(0.6);

        // Suppress whisper.cpp's own stdout/stderr output.
        // All observability goes through `tracing`.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        tracing::debug!(
            model = self.model_name(),
            language = language.unwrap_or("auto"),
            samples = samples.len(),
            "starting inference"
        );

        state
            .full(params, samples)
            .map_err(|e| EngineError::Inference(e.to_string()))?;

        let n_segments = state
            .full_n_segments()
            .map_err(|e| EngineError::SegmentRead(e.to_string()))?;

        let mut parts: Vec<String> = Vec::with_capacity(n_segments as usize);

        for i in 0..n_segments {
            let segment = state
                .full_get_segment_text(i)
                .map_err(|e| EngineError::SegmentRead(format!("segment {i}: {e}")))?;

            let trimmed = segment.trim().to_string();
            if !trimmed.is_empty() {
                parts.push(trimmed);
            }
        }

        let text = parts.join(" ");

        tracing::debug!(
            model = self.model_name(),
            segments = n_segments,
            chars = text.len(),
            "inference complete"
        );

        Ok(TranscriptionOutput { text })
    }
}
