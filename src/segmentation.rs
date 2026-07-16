//! Offline speech segmentation — Silero-style timestamp extraction.
//!
//! Port of FluidAudio's `VadManager+SpeechSegmentation.swift`
//! (`detectSpeechSampleRanges`), semantics preserved exactly; the only
//! difference is frame granularity (512-sample hop here vs 4096 there).

use crate::model::{FluidVadError, SileroModel, FRAME_SIZE, SAMPLE_RATE};

/// Segmentation behavior configuration. Defaults mirror FluidAudio's
/// `VadSegmentationConfig`, except `threshold`: FluidAudio defaults to 0.85
/// (tuned for its CoreML conversion); we run the upstream ONNX weights, so we
/// keep upstream Silero's 0.5.
#[derive(Clone, Debug)]
pub struct VadSegmentationConfig {
    /// Entry threshold: a frame with probability >= this starts/continues speech.
    pub threshold: f32,
    /// Minimum speech run to keep, seconds.
    pub min_speech_duration: f64,
    /// Silence needed to close a segment, seconds.
    pub min_silence_duration: f64,
    /// Split segments longer than this, seconds (`f64::INFINITY` disables).
    pub max_speech_duration: f64,
    /// Padding added around each segment, seconds.
    pub speech_padding: f64,
    /// A candidate split silence must dip below this probability to be preferred.
    pub silence_threshold_for_split: f32,
    /// Pin the exit threshold; `None` derives it from `threshold - negative_threshold_offset`.
    pub negative_threshold: Option<f32>,
    /// Offset used to derive the exit threshold when `negative_threshold` is `None`.
    pub negative_threshold_offset: f32,
    /// Minimum silence considered a split candidate once a segment exceeds max duration, seconds.
    pub min_silence_at_max_speech: f64,
    /// When no candidate dips below `silence_threshold_for_split`, split at the
    /// longest candidate silence (`true`) or the most recent one (`false`).
    pub use_max_possible_silence_at_max_speech: bool,
}

impl Default for VadSegmentationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration: 0.15,
            min_silence_duration: 0.75,
            max_speech_duration: 14.0,
            speech_padding: 0.1,
            silence_threshold_for_split: 0.3,
            negative_threshold: None,
            negative_threshold_offset: 0.15,
            min_silence_at_max_speech: 0.098,
            use_max_possible_silence_at_max_speech: true,
        }
    }
}

impl VadSegmentationConfig {
    /// Entry threshold. Always `threshold`.
    ///
    /// Deviation from FluidAudio: Swift derives the entry threshold from a
    /// pinned negative (`negative + offset`) because its entry threshold lives
    /// on a different struct. Here both are fields of this config, so silently
    /// discarding a user-set `threshold` would be a footgun — pinning
    /// `negative_threshold` only pins the exit threshold.
    pub fn effective_threshold(&self) -> f32 {
        self.threshold
    }

    /// Working exit threshold for hysteresis (Silero heuristic, override escape hatch).
    pub fn effective_negative_threshold(&self) -> f32 {
        match self.negative_threshold {
            Some(v) => v,
            None => (self.threshold - self.negative_threshold_offset).max(0.01),
        }
    }

    /// Validate ranges (mirrors FluidAudio's preconditions), rejecting NaN.
    pub fn validate(&self) -> Result<(), String> {
        let in_unit = |v: f32| !v.is_nan() && (0.0..=1.0).contains(&v);
        if !in_unit(self.threshold) {
            return Err(format!(
                "threshold must be in [0, 1], got {}",
                self.threshold
            ));
        }
        if let Some(n) = self.negative_threshold {
            if !in_unit(n) {
                return Err(format!("negativeThreshold must be in [0, 1], got {n}"));
            }
            if n > self.threshold {
                return Err(format!(
                    "negativeThreshold ({n}) must not exceed threshold ({}) — hysteresis inverts otherwise",
                    self.threshold
                ));
            }
        }
        if self.negative_threshold_offset.is_nan() || self.negative_threshold_offset < 0.0 {
            return Err("negativeThresholdOffset must be non-negative".into());
        }
        if !in_unit(self.silence_threshold_for_split) {
            return Err("silenceThresholdForSplit must be in [0, 1]".into());
        }
        for (name, v) in [
            ("minSpeechDuration", self.min_speech_duration),
            ("minSilenceDuration", self.min_silence_duration),
            ("speechPadding", self.speech_padding),
            ("minSilenceAtMaxSpeech", self.min_silence_at_max_speech),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(format!(
                    "{name} must be a non-negative finite number, got {v}"
                ));
            }
        }
        // NaN and non-positive are rejected; +inf is allowed (disables splitting)
        if self.max_speech_duration.is_nan() || self.max_speech_duration <= 0.0 {
            return Err(format!(
                "maxSpeechDuration must be positive (or infinite to disable), got {}",
                self.max_speech_duration
            ));
        }
        Ok(())
    }
}

/// A detected speech region, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadSegment {
    pub start_time: f64,
    pub end_time: f64,
}

impl VadSegment {
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    pub fn start_sample(&self) -> usize {
        (self.start_time * SAMPLE_RATE as f64) as usize
    }

    pub fn end_sample(&self) -> usize {
        (self.end_time * SAMPLE_RATE as f64) as usize
    }
}

/// Segment a 16 kHz mono buffer into speech regions.
pub fn segment_speech(
    model: &SileroModel,
    samples: &[f32],
    config: &VadSegmentationConfig,
) -> Result<Vec<VadSegment>, FluidVadError> {
    let probabilities = model.probabilities(samples)?;
    Ok(segment_from_probabilities(
        &probabilities,
        samples.len(),
        config,
    ))
}

/// Run the segmentation state machine over precomputed frame probabilities
/// (one per [`FRAME_SIZE`] hop). Exposed for testing with synthetic sequences.
pub fn segment_from_probabilities(
    probabilities: &[f32],
    total_samples: usize,
    config: &VadSegmentationConfig,
) -> Vec<VadSegment> {
    if probabilities.is_empty() || total_samples == 0 {
        return vec![];
    }
    detect_speech_sample_ranges(probabilities, total_samples, config)
        .into_iter()
        .map(|(start, end)| {
            let start = start.min(total_samples);
            let end = end.clamp(start, total_samples);
            VadSegment {
                start_time: start as f64 / SAMPLE_RATE as f64,
                end_time: end as f64 / SAMPLE_RATE as f64,
            }
        })
        .collect()
}

struct CandidateSilence {
    start: usize,
    duration: usize,
    min_probability: f32,
}

fn detect_speech_sample_ranges(
    probabilities: &[f32],
    audio_length_samples: usize,
    config: &VadSegmentationConfig,
) -> Vec<(usize, usize)> {
    let hop_size = FRAME_SIZE;
    let window_size = FRAME_SIZE;
    let sr = SAMPLE_RATE as f64;
    let threshold = config.effective_threshold();
    let negative_threshold = config.effective_negative_threshold();
    let min_speech_samples = (config.min_speech_duration * sr) as usize;
    let speech_pad_samples = (config.speech_padding * sr) as usize;
    let max_speech_samples = if config.max_speech_duration.is_infinite() {
        usize::MAX
    } else {
        ((config.max_speech_duration * sr) as i64 - (window_size + 2 * speech_pad_samples) as i64)
            .max(0) as usize
    };
    let min_silence_samples = (config.min_silence_duration * sr) as usize;
    let min_silence_at_max_speech = (config.min_silence_at_max_speech * sr) as usize;

    let mut triggered = false;
    let mut current_speech_start: usize = 0;
    let mut temp_end: Option<usize> = None;
    let mut temp_silence_min_prob: Option<f32> = None;
    let mut possible_ends: Vec<CandidateSilence> = Vec::new();
    let mut speeches: Vec<(usize, usize)> = Vec::new();

    let flush = |speeches: &mut Vec<(usize, usize)>, start: usize, end: usize| {
        if end > start && (end - start) >= min_speech_samples {
            speeches.push((start, end.min(audio_length_samples)));
        }
    };

    for (index, &prob) in probabilities.iter().enumerate() {
        let frame_start = index * hop_size;

        if prob >= threshold {
            if let Some(temp_end_sample) = temp_end {
                let silence_duration = frame_start - temp_end_sample;
                if silence_duration > min_silence_at_max_speech {
                    possible_ends.push(CandidateSilence {
                        start: temp_end_sample,
                        duration: silence_duration,
                        min_probability: temp_silence_min_prob.unwrap_or(1.0),
                    });
                }
            }
            temp_end = None;
            temp_silence_min_prob = None;

            if !triggered {
                triggered = true;
                current_speech_start = frame_start;
                continue;
            }
        }

        if triggered && max_speech_samples < usize::MAX {
            let current_duration = frame_start.saturating_sub(current_speech_start);
            if current_duration > max_speech_samples {
                let chosen_split: Option<&CandidateSilence> = if possible_ends.is_empty() {
                    None
                } else if let Some(below) = possible_ends
                    .iter()
                    .filter(|c| c.min_probability <= config.silence_threshold_for_split)
                    .max_by_key(|c| c.duration)
                {
                    Some(below)
                } else if config.use_max_possible_silence_at_max_speech {
                    possible_ends.iter().max_by_key(|c| c.duration)
                } else {
                    possible_ends.last()
                };

                let split_end = chosen_split.map(|c| c.start).unwrap_or(frame_start);
                let new_start = chosen_split.map(|c| c.start + c.duration);
                flush(&mut speeches, current_speech_start, split_end);

                triggered = match new_start {
                    Some(new_start) if new_start < frame_start => {
                        current_speech_start = new_start;
                        true
                    }
                    _ => false,
                };

                possible_ends.clear();
                temp_end = None;
                temp_silence_min_prob = None;

                if !triggered {
                    continue;
                }
            }
        }

        if prob < negative_threshold && triggered {
            if temp_end.is_none() {
                temp_end = Some(frame_start);
            }
            temp_silence_min_prob = Some(match temp_silence_min_prob {
                Some(v) => v.min(prob),
                None => prob,
            });
            if let Some(start_silence) = temp_end {
                if frame_start - start_silence >= min_silence_samples {
                    flush(&mut speeches, current_speech_start, start_silence);
                    triggered = false;
                    temp_end = None;
                    temp_silence_min_prob = None;
                    possible_ends.clear();
                    continue;
                }
            }
        }
    }

    if triggered {
        flush(&mut speeches, current_speech_start, audio_length_samples);
    }

    if speeches.is_empty() {
        return vec![];
    }

    // Padding pass: pad edges, split inter-segment silences shorter than 2*pad.
    let mut adjusted = speeches;
    let n = adjusted.len();
    for index in 0..n {
        if index == 0 {
            adjusted[index].0 = adjusted[index].0.saturating_sub(speech_pad_samples);
        }
        if index < n - 1 {
            let silence = adjusted[index + 1].0.saturating_sub(adjusted[index].1);
            if silence < 2 * speech_pad_samples {
                let half = silence / 2;
                adjusted[index].1 = (adjusted[index].1 + half).min(audio_length_samples);
                adjusted[index + 1].0 = adjusted[index + 1].0.saturating_sub(half);
            } else {
                adjusted[index].1 =
                    (adjusted[index].1 + speech_pad_samples).min(audio_length_samples);
                adjusted[index + 1].0 = adjusted[index + 1].0.saturating_sub(speech_pad_samples);
            }
        } else {
            adjusted[index].1 = (adjusted[index].1 + speech_pad_samples).min(audio_length_samples);
        }
    }

    adjusted
        .into_iter()
        .map(|(start, end)| {
            let start = start.min(audio_length_samples);
            let end = end.clamp(start, audio_length_samples);
            (start, end)
        })
        .filter(|(start, end)| end > start)
        .collect()
}
