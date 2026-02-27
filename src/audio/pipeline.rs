use std::path::Path;

use super::decoder::{self, DecodeError, SUPPORTED_EXTENSIONS};
use super::resampler::{self, ResampleError};

/// Limits injected by the caller (sourced from `config::Config`).
/// Keeping them as an explicit parameter — rather than reading env vars here —
/// ensures the `audio` module has zero external dependencies and stays testable.
///
/// `Copy` is derived so the value can be moved into `tokio::task::spawn_blocking`
/// closures without an explicit clone.
#[derive(Debug, Clone, Copy)]
pub struct AudioLimits {
    pub max_file_size_bytes: u64,
    pub max_duration_secs: f64,
}

/// Mono 16 kHz f32 samples ready for whisper.cpp, together with the audio duration.
pub struct ProcessedAudio {
    /// Mono 16 kHz f32 PCM samples.
    pub samples: Vec<f32>,
    /// Audio duration in seconds. Derived from the decoded sample count.
    pub duration_secs: f64,
}

/// All errors that can occur during the audio pipeline.
#[derive(Debug)]
pub enum AudioError {
    FileNotFound(String),
    UnsupportedFormat(String),
    FileTooLarge { size_bytes: u64, limit_bytes: u64 },
    DurationExceeded { duration_secs: f64, limit_secs: f64 },
    Decode(DecodeError),
    Resample(ResampleError),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(p) => {
                write!(f, "audio file not found: {p}")
            }
            Self::UnsupportedFormat(ext) => {
                write!(
                    f,
                    "unsupported audio format: .{ext}. Supported: {}",
                    SUPPORTED_EXTENSIONS.join(", ")
                )
            }
            Self::FileTooLarge { size_bytes, limit_bytes } => write!(
                f,
                "file size ({:.2} MB) exceeds limit ({:.2} MB)",
                *size_bytes as f64 / 1_048_576.0,
                *limit_bytes as f64 / 1_048_576.0,
            ),
            Self::DurationExceeded { duration_secs, limit_secs } => write!(
                f,
                "audio duration ({duration_secs:.2}s) exceeds limit ({limit_secs:.2}s)",
            ),
            Self::Decode(e) => write!(f, "decode error: {e}"),
            Self::Resample(e) => write!(f, "resample error: {e}"),
        }
    }
}

impl std::error::Error for AudioError {}

// Transparent conversions from inner error types.
// FileNotFound and UnsupportedFormat are promoted to top-level variants
// so callers get clean, specific errors without unwrapping nested types.
impl From<DecodeError> for AudioError {
    fn from(e: DecodeError) -> Self {
        match e {
            DecodeError::FileNotFound(p) => Self::FileNotFound(p),
            DecodeError::UnsupportedFormat(ext) => Self::UnsupportedFormat(ext),
            e => Self::Decode(e),
        }
    }
}

impl From<ResampleError> for AudioError {
    fn from(e: ResampleError) -> Self {
        Self::Resample(e)
    }
}

/// Full audio processing pipeline: validate → decode → check duration → resample.
///
/// Returns [`ProcessedAudio`] containing mono 16 kHz f32 samples ready for
/// whisper.cpp inference, together with the audio duration needed for the result message.
///
/// # Validation order
/// Mirrors the original Python `AudioProcessor.process_audio`:
/// 1. File existence + size (`validate_file_size`)
/// 2. Extension (`validate_extension`)
/// 3. Decode with symphonia
/// 4. Duration check (accurate: derived from decoded sample count)
/// 5. Resample to mono 16 kHz
///
/// # Errors
/// Returns [`AudioError`] for any validation or processing failure.
pub fn process(path: &Path, limits: &AudioLimits) -> Result<ProcessedAudio, AudioError> {
    // 1. Existence + file size
    validate_file_size(path, limits.max_file_size_bytes)?;

    // 2. Extension (fast check before opening the file with the full decoder)
    validate_extension(path)?;

    // 3. Decode
    let decoded = decoder::decode(path)?;

    // 4. Duration — checked after decoding so the count is derived from actual samples,
    //    not from container metadata (which can be wrong or absent for VBR files).
    if decoded.duration_secs > limits.max_duration_secs {
        return Err(AudioError::DurationExceeded {
            duration_secs: decoded.duration_secs,
            limit_secs: limits.max_duration_secs,
        });
    }

    // 5. Resample to mono 16 kHz
    let duration_secs = decoded.duration_secs;
    let samples = resampler::to_mono_16k(&decoded)?;

    Ok(ProcessedAudio { samples, duration_secs })
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn validate_file_size(path: &Path, max_bytes: u64) -> Result<(), AudioError> {
    let metadata = std::fs::metadata(path)
        .map_err(|_| AudioError::FileNotFound(path.display().to_string()))?;

    let size = metadata.len();

    if size > max_bytes {
        return Err(AudioError::FileTooLarge {
            size_bytes: size,
            limit_bytes: max_bytes,
        });
    }

    Ok(())
}

fn validate_extension(path: &Path) -> Result<(), AudioError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();

    if !SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
        return Err(AudioError::UnsupportedFormat(ext));
    }

    Ok(())
}
