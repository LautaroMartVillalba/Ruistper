use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use super::decoder::DecodedAudio;

/// Sample rate required by whisper.cpp.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug)]
pub enum ResampleError {
    Construction(String),
    Processing(String),
}

impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Construction(msg) => write!(f, "resampler construction failed: {msg}"),
            Self::Processing(msg) => write!(f, "resampling failed: {msg}"),
        }
    }
}

impl std::error::Error for ResampleError {}

/// Convert decoded audio to a mono f32 vector at 16 kHz — the exact input format
/// expected by whisper.cpp.
///
/// # Steps
/// 1. Mix all channels to mono by arithmetic mean.
/// 2. If `decoded.sample_rate` is already 16 000 Hz, return directly.
/// 3. Otherwise resample using a sinc interpolator (rubato `SincFixedIn`).
pub fn to_mono_16k(decoded: &DecodedAudio) -> Result<Vec<f32>, ResampleError> {
    let mono = mix_to_mono(&decoded.samples, decoded.channels as usize);

    if decoded.sample_rate == TARGET_SAMPLE_RATE {
        return Ok(mono);
    }

    resample(mono, decoded.sample_rate, TARGET_SAMPLE_RATE)
}

/// Average interleaved multi-channel samples into a single mono channel.
/// For already-mono input this is a zero-cost copy.
fn mix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }

    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample a mono f32 buffer from `from_rate` Hz to `to_rate` Hz.
///
/// Uses [`SincFixedIn`] which requires a fixed number of input frames per call
/// (`input_frames_next()`). The last chunk is zero-padded to meet this requirement,
/// and the resampler is flushed with `process_partial` to drain internal latency.
fn resample(samples: Vec<f32>, from_rate: u32, to_rate: u32) -> Result<Vec<f32>, ResampleError> {
    let ratio = to_rate as f64 / from_rate as f64;

    // Quality-vs-speed parameters. `sinc_len=256` and `oversampling_factor=256`
    // give high quality at the cost of a slightly larger filter; acceptable for
    // batch audio processing where latency is not interactive.
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // Chunk size of 1024 frames balances per-call overhead against memory pressure.
    let chunk_size = 1024_usize;

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .map_err(|e| ResampleError::Construction(e.to_string()))?;

    let expected_output = (samples.len() as f64 * ratio) as usize + chunk_size;
    let mut output: Vec<f32> = Vec::with_capacity(expected_output);

    let mut pos = 0;
    while pos < samples.len() {
        // `input_frames_next()` always returns `chunk_size` for SincFixedIn,
        // but we ask explicitly to stay correct under any resampler implementation.
        let needed = resampler.input_frames_next();
        let end = (pos + needed).min(samples.len());

        let mut chunk = samples[pos..end].to_vec();
        // Zero-pad the last (possibly short) chunk to meet the fixed-size requirement.
        chunk.resize(needed, 0.0);

        let waves_out = resampler
            .process(&[chunk], None)
            .map_err(|e| ResampleError::Processing(e.to_string()))?;

        output.extend_from_slice(&waves_out[0]);
        pos += needed;
    }

    // Flush samples held inside the resampler's internal delay line.
    let tail = resampler
        .process_partial::<Vec<f32>>(None, None)
        .map_err(|e| ResampleError::Processing(e.to_string()))?;

    if let Some(tail_channel) = tail.first() {
        output.extend_from_slice(tail_channel);
    }

    Ok(output)
}
