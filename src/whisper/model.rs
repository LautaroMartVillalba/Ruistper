use std::path::Path;

use whisper_rs::{WhisperContext, WhisperContextParameters};

/// A GGML Whisper model loaded into memory.
///
/// # Memory lifecycle
/// Wrap in `Arc<WhisperModel>` and pass clones to each worker.
/// As long as any `Arc` clone is alive the model weights stay resident in RAM —
/// no reload occurs when individual workers restart or are rebuilt.
///
/// # Thread safety
/// `WhisperContext` is `Send + Sync`: the model weights are immutable after loading
/// and whisper.cpp's context is safe to read concurrently from multiple threads.
/// Mutable inference state lives in `WhisperState`, which is created per-call
/// inside [`super::engine::WhisperEngine::transcribe`].
pub struct WhisperModel {
    /// whisper.cpp context. Immutable after construction.
    pub(super) context: WhisperContext,

    /// Human-readable model identifier returned in result messages (e.g. `"base.q5_0"`).
    pub name: String,
}

// Safety: whisper.cpp contexts are safe to send and share across threads.
// The context owns the model weights and exposes only read operations after init.
unsafe impl Send for WhisperModel {}
unsafe impl Sync for WhisperModel {}

/// Errors that can occur while loading the model.
#[derive(Debug)]
pub enum ModelError {
    /// The model file does not exist at the given path.
    FileNotFound(String),
    /// whisper.cpp rejected the model file (wrong format, corrupt, etc.).
    Load(String),
    /// The path contains non-UTF-8 characters, which whisper.cpp cannot handle.
    InvalidPath(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "model file not found: {p}"),
            Self::Load(msg) => write!(f, "failed to load whisper model: {msg}"),
            Self::InvalidPath(p) => {
                write!(f, "model path is not valid UTF-8: {p}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

impl WhisperModel {
    /// Load a GGML model file from disk into memory.
    ///
    /// This is expensive — it reads the full model weights. Call **once** at startup
    /// and distribute via `Arc<WhisperModel>`.
    ///
    /// # Arguments
    /// * `model_path` — absolute path to the `.bin` GGML model file
    ///   (e.g. `/app/models/ggml-base.q5_0.bin`).
    /// * `model_name` — label stored in the model and returned in result messages
    ///   (e.g. `"base.q5_0"`). Does not need to match the filename.
    ///
    /// # Errors
    /// Returns [`ModelError`] if the file is missing, not UTF-8, or rejected by whisper.cpp.
    pub fn load(model_path: &Path, model_name: String) -> Result<Self, ModelError> {
        if !model_path.exists() {
            return Err(ModelError::FileNotFound(model_path.display().to_string()));
        }

        let path_str = model_path.to_str().ok_or_else(|| {
            ModelError::InvalidPath(model_path.display().to_string())
        })?;

        tracing::info!(model = %model_name, path = %path_str, "loading whisper model");

        let context = WhisperContext::new_with_params(
            path_str,
            WhisperContextParameters::default(),
        )
        .map_err(|e| ModelError::Load(e.to_string()))?;

        tracing::info!(model = %model_name, "whisper model loaded");

        Ok(Self {
            context,
            name: model_name,
        })
    }
}
