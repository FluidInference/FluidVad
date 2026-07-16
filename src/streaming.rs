//! Streaming VAD with start/end events — Silero-style hysteresis with
//! min-speech blip suppression and max-speech force-splitting.

use crate::model::{FluidVadError, ModelState, SileroModel, FRAME_SIZE, SAMPLE_RATE};
use crate::segmentation::VadSegmentationConfig;

/// Event kind emitted by the streaming state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadStreamEventKind {
    SpeechStart,
    SpeechEnd,
}

/// A speech boundary event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadStreamEvent {
    pub kind: VadStreamEventKind,
    /// Sample index the boundary refers to (padding already applied).
    /// u64: a continuous 16 kHz stream overflows u32 after ~74 hours, which is
    /// a reachable session length on the wasm32 target where usize is 32-bit.
    pub sample_index: u64,
}

impl VadStreamEvent {
    pub fn time_seconds(&self) -> f64 {
        self.sample_index as f64 / SAMPLE_RATE as f64
    }

    pub fn is_start(&self) -> bool {
        self.kind == VadStreamEventKind::SpeechStart
    }

    pub fn is_end(&self) -> bool {
        self.kind == VadStreamEventKind::SpeechEnd
    }
}

/// Streaming hysteresis state, carried between frames.
#[derive(Clone, Debug, Default)]
pub struct VadStreamState {
    pub(crate) model_state: ModelState,
    pub triggered: bool,
    pub temp_end_sample: Option<u64>,
    pub processed_samples: u64,
    /// Start of a speech run that hasn't persisted `min_speech_duration` yet.
    pub pending_start: Option<u64>,
    /// Start of the confirmed speech run (for `max_speech_duration` splits).
    pub speech_start_sample: u64,
}

impl VadStreamState {
    /// Fresh state, mirroring Silero's `reset_states`.
    pub fn initial() -> Self {
        Self::default()
    }
}

/// Result of processing one frame.
#[derive(Clone, Copy, Debug)]
pub struct VadStreamFrameResult {
    pub probability: f32,
    pub event: Option<VadStreamEvent>,
}

/// Convenience streaming wrapper: owns the model state and an internal buffer
/// so callers can push chunks of any length; emits one result per full frame.
pub struct VadStreamer {
    model: SileroModel,
    config: VadSegmentationConfig,
    state: VadStreamState,
    pending: Vec<f32>,
}

impl VadStreamer {
    pub fn new(config: VadSegmentationConfig) -> Result<Self, FluidVadError> {
        Ok(Self {
            model: SileroModel::new()?,
            config,
            state: VadStreamState::initial(),
            pending: Vec::new(),
        })
    }

    pub fn with_model(model: SileroModel, config: VadSegmentationConfig) -> Self {
        Self {
            model,
            config,
            state: VadStreamState::initial(),
            pending: Vec::new(),
        }
    }

    /// Push samples of any length; returns one result per completed 512-sample frame.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<VadStreamFrameResult>, FluidVadError> {
        self.pending.extend_from_slice(samples);
        let mut results = Vec::with_capacity(self.pending.len() / FRAME_SIZE);
        let mut offset = 0;
        let mut error = None;
        while self.pending.len() - offset >= FRAME_SIZE {
            let frame = &self.pending[offset..offset + FRAME_SIZE];
            match self.model.process_frame(frame, &self.state.model_state) {
                Ok((probability, model_state)) => {
                    let event = streaming_state_machine(
                        probability,
                        FRAME_SIZE,
                        model_state,
                        &mut self.state,
                        &self.config,
                    );
                    results.push(VadStreamFrameResult { probability, event });
                    offset += FRAME_SIZE;
                }
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        // drain exactly the consumed prefix — even on error, so a retry never
        // replays frames the model state has already seen
        self.pending.drain(..offset);
        match error {
            Some(e) => Err(e),
            None => Ok(results),
        }
    }

    /// Total samples consumed so far (excludes buffered partial frame).
    pub fn processed_samples(&self) -> u64 {
        self.state.processed_samples
    }

    /// Whether the state machine is currently inside speech.
    pub fn is_speaking(&self) -> bool {
        self.state.triggered
    }

    /// Reset hysteresis + model state (Silero `reset_states`) and drop buffered samples.
    pub fn reset(&mut self) {
        self.state = VadStreamState::initial();
        self.pending.clear();
    }

    /// Borrow the underlying model (e.g. for offline segmentation with the same instance).
    pub fn model(&self) -> &SileroModel {
        &self.model
    }

    /// The configuration this streamer was built with.
    pub fn config(&self) -> &VadSegmentationConfig {
        &self.config
    }
}

/// The pure streaming state machine (Silero-style hysteresis).
///
/// Honors the full config: `min_speech_duration` suppresses blips (a
/// SpeechStart only fires once speech has persisted that long, backdated to
/// the real start), and `max_speech_duration` force-splits long runs so
/// consumers never buffer unboundedly.
pub(crate) fn streaming_state_machine(
    probability: f32,
    chunk_sample_count: usize,
    model_state: ModelState,
    state: &mut VadStreamState,
    config: &VadSegmentationConfig,
) -> Option<VadStreamEvent> {
    state.model_state = model_state;
    state.processed_samples += chunk_sample_count as u64;

    let threshold = config.effective_threshold();
    let negative_threshold = config.effective_negative_threshold();
    let sr = SAMPLE_RATE as f64;
    let chunk = chunk_sample_count as u64;
    let speech_pad_samples = (config.speech_padding * sr) as u64;
    let min_silence_samples = (config.min_silence_duration * sr) as u64;
    let min_speech_samples = (config.min_speech_duration * sr) as u64;
    let max_speech_samples = if config.max_speech_duration.is_infinite() {
        u64::MAX
    } else {
        (config.max_speech_duration * sr) as u64
    };

    // force-split: a confirmed run that reached max_speech_duration ends now;
    // the very same frame becomes a pending start so speech resumes promptly
    if state.triggered && state.processed_samples - state.speech_start_sample >= max_speech_samples
    {
        state.triggered = false;
        state.temp_end_sample = None;
        state.pending_start = Some(state.processed_samples.saturating_sub(chunk));
        return Some(VadStreamEvent {
            kind: VadStreamEventKind::SpeechEnd,
            sample_index: state.processed_samples,
        });
    }

    if probability >= threshold {
        state.temp_end_sample = None;
        if !state.triggered {
            // record where this run began (first frame at/above threshold)
            let run_start = *state
                .pending_start
                .get_or_insert(state.processed_samples.saturating_sub(chunk));
            // confirm once the run has persisted min_speech_duration
            if state.processed_samples - run_start >= min_speech_samples {
                state.triggered = true;
                state.pending_start = None;
                state.speech_start_sample = run_start;
                return Some(VadStreamEvent {
                    kind: VadStreamEventKind::SpeechStart,
                    sample_index: run_start.saturating_sub(speech_pad_samples),
                });
            }
        }
    } else if probability < negative_threshold {
        if state.triggered {
            if state.temp_end_sample.is_none() {
                state.temp_end_sample = Some(state.processed_samples);
            }
            if let Some(silence_start) = state.temp_end_sample {
                if state.processed_samples - silence_start >= min_silence_samples {
                    let raw_end = (silence_start + speech_pad_samples).saturating_sub(chunk);
                    state.triggered = false;
                    state.temp_end_sample = None;
                    return Some(VadStreamEvent {
                        kind: VadStreamEventKind::SpeechEnd,
                        sample_index: raw_end,
                    });
                }
            }
        } else {
            // a pending run that fell back below the exit threshold before
            // reaching min_speech_duration was a blip — drop it silently
            state.pending_start = None;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelState;

    const F: usize = 512; // one frame

    fn drive(probs: &[f32], config: &VadSegmentationConfig) -> Vec<(usize, VadStreamEvent)> {
        let mut state = VadStreamState::initial();
        let mut events = Vec::new();
        for (i, &p) in probs.iter().enumerate() {
            if let Some(e) =
                streaming_state_machine(p, F, ModelState::default(), &mut state, config)
            {
                events.push((i, e));
            }
        }
        events
    }

    #[test]
    fn blip_shorter_than_min_speech_is_suppressed() {
        // default min_speech 0.15s = 2400 samples = 4.7 frames; 3-frame blip
        let mut probs = vec![0.05f32; 10];
        probs.extend([0.95; 3]);
        probs.extend([0.05; 40]);
        let events = drive(&probs, &VadSegmentationConfig::default());
        assert!(events.is_empty(), "{events:?}");
    }

    #[test]
    fn start_fires_backdated_after_min_speech_persists() {
        let mut probs = vec![0.05f32; 10];
        probs.extend([0.95; 20]);
        let events = drive(&probs, &VadSegmentationConfig::default());
        assert_eq!(events.len(), 1);
        let (fired_at, e) = events[0];
        assert!(e.is_start());
        // fired on the 5th speech frame (2560 >= 2400), backdated to the run
        // start (frame 10) minus padding (1600)
        assert_eq!(fired_at, 14);
        assert_eq!(e.sample_index, (10 * F) as u64 - 1600);
    }

    #[test]
    fn zero_min_speech_fires_immediately() {
        let cfg = VadSegmentationConfig {
            min_speech_duration: 0.0,
            ..VadSegmentationConfig::default()
        };
        let mut probs = vec![0.05f32; 4];
        probs.extend([0.95; 2]);
        let events = drive(&probs, &cfg);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, 4, "must fire on the first speech frame");
    }

    #[test]
    fn max_speech_force_splits_long_runs() {
        let cfg = VadSegmentationConfig {
            max_speech_duration: 1.0, // 16000 samples ≈ 31.25 frames
            ..VadSegmentationConfig::default()
        };
        let mut probs = vec![0.95f32; 100];
        probs.extend([0.05; 40]);
        let events = drive(&probs, &cfg);
        // start, forced end, restart, forced end, restart, ..., final silence end
        assert!(events.len() >= 4, "{events:?}");
        let mut expect_start = true;
        for (_, e) in &events {
            assert_eq!(
                e.is_start(),
                expect_start,
                "events must alternate: {events:?}"
            );
            expect_start = !expect_start;
        }
        assert!(!events.last().unwrap().1.is_start(), "must end closed");
    }

    #[test]
    fn hysteresis_band_keeps_pending_and_triggered_runs_alive() {
        // dip into the band (between exit 0.35 and entry 0.5) must not cancel
        // a pending run nor close a confirmed one
        let mut probs = vec![0.95f32; 3]; // pending (needs 5 to confirm)
        probs.extend([0.40; 2]); // band: pending survives
        probs.extend([0.95; 4]); // resumes; run persists from frame 0
        probs.extend([0.05; 40]);
        let events = drive(&probs, &VadSegmentationConfig::default());
        assert!(
            events.first().map(|(_, e)| e.is_start()).unwrap_or(false),
            "{events:?}"
        );
    }
}
