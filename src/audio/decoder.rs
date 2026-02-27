use std::fs::File;
use std::io::Read;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Raw decoded audio — interleaved f32 PCM, original sample rate and channel count.
/// No resampling or channel mixing is applied here; that is delegated to `resampler`.
pub struct DecodedAudio {
    /// Interleaved f32 PCM samples across all channels.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
    /// Duration derived from sample count (accurate for CBR; approximate for VBR).
    pub duration_secs: f64,
}

#[derive(Debug)]
pub enum DecodeError {
    FileNotFound(String),
    UnsupportedFormat(String),
    Io(std::io::Error),
    Failed(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "file not found: {p}"),
            Self::UnsupportedFormat(ext) => write!(f, "unsupported format: .{ext}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Failed(msg) => write!(f, "decode failed: {msg}"),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DecodeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Audio file extensions accepted by the pipeline.
/// `.wma` is excluded — not supported by symphonia.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["opus", "mp3", "wav", "m4a", "ogg", "flac", "aac"];

/// Decode an audio file at `path` to raw interleaved f32 PCM samples.
///
/// Delegates format detection and decoding entirely to symphonia.
/// The returned [`DecodedAudio`] preserves the original sample rate and channel layout;
/// normalization to mono 16 kHz is handled by [`super::resampler::to_mono_16k`].
///
/// # Errors
/// - [`DecodeError::FileNotFound`] if `path` does not exist.
/// - [`DecodeError::UnsupportedFormat`] if the file extension is not in [`SUPPORTED_EXTENSIONS`].
/// - [`DecodeError::Failed`] on any symphonia probe or decode error.
pub fn decode(path: &Path) -> Result<DecodedAudio, DecodeError> {
    if !path.exists() {
        return Err(DecodeError::FileNotFound(path.display().to_string()));
    }

    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();

    if !SUPPORTED_EXTENSIONS.contains(&extension.as_str()) {
        return Err(DecodeError::UnsupportedFormat(extension));
    }

    // Symphonia 0.5 has no Opus codec. OGG files from WhatsApp are OGG/Opus.
    // Peek at the first 64 bytes: every valid OGG Opus stream has "OpusHead"
    // within the first OGG page (before any audio data).
    if extension == "ogg" || extension == "opus" {
        if is_ogg_opus(path)? {
            return decode_ogg_opus(path);
        }
    }

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(&extension);

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| DecodeError::Failed(e.to_string()))?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| DecodeError::Failed("no default audio track found".to_string()))?;

    let track_id = track.id;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| DecodeError::Failed("sample rate missing from codec params".to_string()))?;

    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u32)
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| DecodeError::Failed(e.to_string()))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // In symphonia 0.5, end-of-stream is signalled via IoError(UnexpectedEof).
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            // ResetRequired means the decoder must be reset; treat as end of usable data.
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(DecodeError::Failed(e.to_string())),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Soft decode errors (e.g., a malformed frame): skip and continue.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(DecodeError::Failed(e.to_string())),
        };

        let spec = *decoded.spec();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(buf.samples());
    }

    let duration_secs = if sample_rate > 0 && channels > 0 {
        all_samples.len() as f64 / (sample_rate as f64 * channels as f64)
    } else {
        0.0
    };

    Ok(DecodedAudio {
        samples: all_samples,
        sample_rate,
        channels,
        duration_secs,
    })
}

// ── OGG/Opus helpers ──────────────────────────────────────────────────────────

/// Return `true` if the file at `path` is an OGG container carrying an Opus stream.
///
/// Every valid OGG Opus stream begins its first logical page with the
/// identification header packet, whose first 8 bytes are `OpusHead`.
/// Scanning the first 64 bytes of the file is sufficient because the OGG
/// page header is at most 27 + 255 bytes, and the first packet starts
/// immediately after — `OpusHead` is always within the first 300 bytes.
fn is_ogg_opus(path: &Path) -> Result<bool, DecodeError> {
    let mut file = File::open(path)?;
    let mut buf = [0u8; 128];
    let n = file.read(&mut buf)?;
    Ok(buf[..n].windows(8).any(|w| w == b"OpusHead"))
}

/// Decode an OGG/Opus file to raw interleaved f32 PCM at 48 kHz.
///
/// Uses the `ogg` crate for demuxing and `opus` (libopus) for decoding.
/// Opus always outputs at 48 kHz; resampling to 16 kHz is done downstream
/// by [`super::resampler::to_mono_16k`].
fn decode_ogg_opus(path: &Path) -> Result<DecodedAudio, DecodeError> {
    use ogg::reading::PacketReader;

    // Opus spec: all decoders output at 48 kHz.
    const OPUS_RATE: u32 = 48_000;
    // Maximum frame size: 120 ms at 48 kHz per channel.
    const MAX_FRAME_SAMPLES: usize = 5_760;

    let file = File::open(path)?;
    let mut reader = PacketReader::new(file);

    let mut channels: u32 = 2;
    let mut decoder: Option<opus::Decoder> = None;
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let pkt = reader
            .read_packet()
            .map_err(|e| DecodeError::Failed(format!("ogg read: {e}")))?;

        let pkt = match pkt {
            Some(p) => p,
            None => break, // end of stream
        };

        let data = &pkt.data;

        // ── Identification header ──────────────────────────────────────────
        if data.starts_with(b"OpusHead") {
            // Byte 9 (0-indexed) is the channel count.
            channels = *data.get(9).unwrap_or(&2) as u32;
            let ch = if channels == 1 {
                opus::Channels::Mono
            } else {
                opus::Channels::Stereo
            };
            decoder = Some(
                opus::Decoder::new(OPUS_RATE, ch)
                    .map_err(|e| DecodeError::Failed(format!("opus init: {e}")))?,
            );
            continue;
        }

        // ── Comment header — skip ──────────────────────────────────────────
        if data.starts_with(b"OpusTags") {
            continue;
        }

        // ── Audio packet ──────────────────────────────────────────────────
        if let Some(ref mut dec) = decoder {
            let mut out = vec![0i16; MAX_FRAME_SAMPLES * channels as usize];
            let samples_per_channel = dec
                .decode(data, &mut out, false)
                .map_err(|e| DecodeError::Failed(format!("opus decode: {e}")))?
                as usize;

            let total = samples_per_channel * channels as usize;
            all_samples.extend(out[..total].iter().map(|&s| s as f32 / 32_768.0));
        }
    }

    if all_samples.is_empty() {
        return Err(DecodeError::Failed(
            "opus: no audio samples decoded".to_string(),
        ));
    }

    let duration_secs =
        all_samples.len() as f64 / (OPUS_RATE as f64 * channels as f64);

    Ok(DecodedAudio {
        samples: all_samples,
        sample_rate: OPUS_RATE,
        channels,
        duration_secs,
    })
}
