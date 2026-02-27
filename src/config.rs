use std::env;
use std::path::PathBuf;

// ── Error ──────────────────────────────────────────────────────────────────────

/// Errors that can occur while loading configuration.
#[derive(Debug)]
pub enum ConfigError {
    /// An environment variable contained an unparseable value.
    Parse {
        var: &'static str,
        raw: String,
        expected: &'static str,
    },
    /// A value was parsed successfully but violated a business-rule constraint.
    InvalidValue {
        var: &'static str,
        message: String,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse { var, raw, expected } => {
                write!(f, "env {var}={raw:?} — expected {expected}")
            }
            Self::InvalidValue { var, message } => {
                write!(f, "env {var}: {message}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

// ── Config ─────────────────────────────────────────────────────────────────────

/// Centralised application configuration.
///
/// All fields are populated from environment variables with hardcoded defaults.
/// Call [`Config::load`] once at startup — it validates every value eagerly so
/// any misconfiguration is reported before any connection attempt is made.
#[derive(Debug, Clone)]
pub struct Config {
    // ── RabbitMQ ──────────────────────────────────────────────────────────────
    /// Full AMQP connection URL.
    /// Env: `RABBITMQ_URL` · Default: `amqp://guest:guest@localhost:5672/`
    pub rabbitmq_url: String,

    /// Logical queue name.  Informational — the topology uses constants in
    /// `messaging::rabbit`, but this field ensures the value is traceable.
    /// Env: `RABBITMQ_QUEUE_NAME` · Default: `whisper_transcriptions`
    pub rabbitmq_queue_name: String,

    // ── Worker pool ───────────────────────────────────────────────────────────
    /// Number of concurrent transcription workers.
    /// Env: `WORKERS_COUNT` · Default: `4` · Constraint: ≥ 1
    pub workers_count: usize,

    // ── Whisper / model ───────────────────────────────────────────────────────
    /// GGML model identifier, used to derive [`Config::model_path`].
    ///
    /// Env: `WHISPER_MODEL` · Default: `base`
    ///
    /// Examples: `tiny`, `base`, `small`, `medium`, `large-v3`, `base.q5_0`
    ///
    /// Model file will be looked up as `{models_dir}/ggml-{whisper_model}.bin`.
    pub whisper_model: String,

    /// Inference device passed to whisper.cpp.
    /// Env: `WHISPER_DEVICE` · Default: `cpu`
    /// Accepted by whisper.cpp: `cpu`, `cuda`, `metal` (platform-dependent).
    pub whisper_device: String,

    /// Compute type. **Not used by whisper.cpp** (faster-whisper / CTranslate2
    /// parameter). Kept for documentation and forward compatibility.
    /// Env: `WHISPER_COMPUTE_TYPE` · Default: `int8`
    pub whisper_compute_type: String,

    /// Directory where GGML `.bin` model files are stored.
    /// Env: `MODELS_DIR` · Default: `/app/models`
    pub models_dir: PathBuf,

    // ── Audio processing ──────────────────────────────────────────────────────
    /// Maximum accepted input file size in megabytes.
    /// Env: `MAX_FILE_SIZE_MB` · Default: `100` · Constraint: ≥ 1
    pub max_file_size_mb: u64,

    /// Maximum accepted audio duration in seconds.
    /// Env: `MAX_AUDIO_DURATION_SEC` · Default: `3600` · Constraint: > 0
    pub max_audio_duration_sec: f64,

    /// Target sample rate for resampling.
    /// Env: `AUDIO_SAMPLE_RATE` · Default: `16000`
    ///
    /// ⚠  whisper.cpp **requires exactly 16 000 Hz**. Any other value is
    ///    rejected at load time.
    pub audio_sample_rate: u32,

    /// Directory for temporary WAV files created during audio processing.
    /// Will be created at startup if it does not exist.
    /// Env: `TMP_DIR` · Default: `/tmp/whisper`
    pub tmp_dir: PathBuf,

    // ── HTTP API ──────────────────────────────────────────────────────────────
    /// Bind host for the HTTP API server.
    /// Env: `API_HOST` · Default: `0.0.0.0`
    pub api_host: String,

    /// Bind port for the HTTP API server.
    /// Env: `API_PORT` · Default: `7050` · Constraint: 1–65535
    pub api_port: u16,
}

impl Config {
    /// Load and validate configuration from environment variables.
    ///
    /// Missing variables fall back to hardcoded defaults.
    /// Returns [`ConfigError`] on the first invalid value encountered.
    pub fn load() -> Result<Self, ConfigError> {
        // ── RabbitMQ ──────────────────────────────────────────────────────────
        let rabbitmq_url =
            env_str("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/");
        let rabbitmq_queue_name =
            env_str("RABBITMQ_QUEUE_NAME", "whisper_transcriptions");

        // ── Worker pool ───────────────────────────────────────────────────────
        let workers_count = parse_usize("WORKERS_COUNT", 4)?;
        validate(
            "WORKERS_COUNT",
            workers_count >= 1,
            "must be ≥ 1",
        )?;

        // ── Whisper / model ───────────────────────────────────────────────────
        let whisper_model = env_str("WHISPER_MODEL", "base");
        validate(
            "WHISPER_MODEL",
            !whisper_model.is_empty(),
            "must not be empty",
        )?;

        let whisper_device = env_str("WHISPER_DEVICE", "cpu");
        let whisper_compute_type = env_str("WHISPER_COMPUTE_TYPE", "int8");
        let models_dir = PathBuf::from(env_str("MODELS_DIR", "/app/models"));

        // ── Audio ─────────────────────────────────────────────────────────────
        let max_file_size_mb = parse_u64("MAX_FILE_SIZE_MB", 100)?;
        validate("MAX_FILE_SIZE_MB", max_file_size_mb >= 1, "must be ≥ 1")?;

        let max_audio_duration_sec = parse_f64("MAX_AUDIO_DURATION_SEC", 3600.0)?;
        validate(
            "MAX_AUDIO_DURATION_SEC",
            max_audio_duration_sec > 0.0,
            "must be > 0",
        )?;

        let audio_sample_rate = parse_u32("AUDIO_SAMPLE_RATE", 16_000)?;
        validate(
            "AUDIO_SAMPLE_RATE",
            audio_sample_rate == 16_000,
            "whisper.cpp requires exactly 16000 Hz",
        )?;

        let tmp_dir = PathBuf::from(env_str("TMP_DIR", "/tmp/whisper"));

        // ── HTTP API ──────────────────────────────────────────────────────────
        let api_host = env_str("API_HOST", "0.0.0.0");
        let api_port = parse_u16("API_PORT", 7050)?;
        validate("API_PORT", api_port > 0, "must be in range 1–65535")?;

        Ok(Self {
            rabbitmq_url,
            rabbitmq_queue_name,
            workers_count,
            whisper_model,
            whisper_device,
            whisper_compute_type,
            models_dir,
            max_file_size_mb,
            max_audio_duration_sec,
            audio_sample_rate,
            tmp_dir,
            api_host,
            api_port,
        })
    }

    // ── Derived helpers ───────────────────────────────────────────────────────

    /// Absolute path to the GGML model file.
    ///
    /// Convention: `{models_dir}/ggml-{whisper_model}.bin`
    ///
    /// | `WHISPER_MODEL`  | Result                              |
    /// |------------------|-------------------------------------|
    /// | `small`          | `/app/models/ggml-small.bin`        |
    /// | `base.q5_0`      | `/app/models/ggml-base.q5_0.bin`    |
    /// | `large-v3`       | `/app/models/ggml-large-v3.bin`     |
    pub fn model_path(&self) -> PathBuf {
        self.models_dir
            .join(format!("ggml-{}.bin", self.whisper_model))
    }

    /// `max_file_size_mb` converted to bytes, ready for [`crate::audio::pipeline::AudioLimits`].
    pub fn max_file_size_bytes(&self) -> u64 {
        self.max_file_size_mb * 1_024 * 1_024
    }

    /// Human-readable `host:port` string for the HTTP API server.
    pub fn api_addr(&self) -> String {
        format!("{}:{}", self.api_host, self.api_port)
    }

    /// Log a summary of the loaded configuration.
    /// Useful at startup to confirm values from env.
    pub fn log_summary(&self) {
        tracing::info!(
            workers     = self.workers_count,
            model       = %self.whisper_model,
            device      = %self.whisper_device,
            model_path  = %self.model_path().display(),
            max_file_mb = self.max_file_size_mb,
            max_dur_sec = self.max_audio_duration_sec,
            tmp_dir     = %self.tmp_dir.display(),
            api_addr    = %self.api_addr(),
            "⚙️  configuration loaded"
        );
    }
}

// ── Private parse helpers ──────────────────────────────────────────────────────

/// Return the env var value as a `String`, or `default` if unset.
fn env_str(var: &str, default: &str) -> String {
    env::var(var).unwrap_or_else(|_| default.to_string())
}

/// Emit a `ConfigError::InvalidValue` if `condition` is false.
fn validate(var: &'static str, condition: bool, message: &str) -> Result<(), ConfigError> {
    if condition {
        Ok(())
    } else {
        Err(ConfigError::InvalidValue {
            var,
            message: message.to_string(),
        })
    }
}

fn parse_usize(var: &'static str, default: usize) -> Result<usize, ConfigError> {
    match env::var(var) {
        Err(_) => Ok(default),
        Ok(raw) => raw.trim().parse::<usize>().map_err(|_| ConfigError::Parse {
            var,
            raw,
            expected: "unsigned integer",
        }),
    }
}

fn parse_u64(var: &'static str, default: u64) -> Result<u64, ConfigError> {
    match env::var(var) {
        Err(_) => Ok(default),
        Ok(raw) => raw.trim().parse::<u64>().map_err(|_| ConfigError::Parse {
            var,
            raw,
            expected: "unsigned integer",
        }),
    }
}

fn parse_u32(var: &'static str, default: u32) -> Result<u32, ConfigError> {
    match env::var(var) {
        Err(_) => Ok(default),
        Ok(raw) => raw.trim().parse::<u32>().map_err(|_| ConfigError::Parse {
            var,
            raw,
            expected: "unsigned integer (u32)",
        }),
    }
}

fn parse_u16(var: &'static str, default: u16) -> Result<u16, ConfigError> {
    match env::var(var) {
        Err(_) => Ok(default),
        Ok(raw) => raw.trim().parse::<u16>().map_err(|_| ConfigError::Parse {
            var,
            raw,
            expected: "port number (1–65535)",
        }),
    }
}

fn parse_f64(var: &'static str, default: f64) -> Result<f64, ConfigError> {
    match env::var(var) {
        Err(_) => Ok(default),
        Ok(raw) => raw.trim().parse::<f64>().map_err(|_| ConfigError::Parse {
            var,
            raw,
            expected: "decimal number",
        }),
    }
}
