mod model;
mod engine;

pub use model::{WhisperModel, ModelError};
pub use engine::{WhisperEngine, EngineError, TranscriptionOutput};
