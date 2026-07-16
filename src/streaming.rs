//! Streaming VAD with start/end events — Silero-style hysteresis.
//!
//! Port of FluidAudio's `VadManager+Streaming.swift` (`streamingStateMachine`),
//! semantics preserved exactly.

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
    pub sample_index: usize,
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
    pub temp_end_sample: Option<usize>,
    pub processed_samples: usize,
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
        while self.pending.len() - offset >= FRAME_SIZE {
            let frame = &self.pending[offset..offset + FRAME_SIZE];
            let (probability, model_state) =
                self.model.process_frame(frame, &self.state.model_state)?;
            let (next, event) = streaming_state_machine(
                probability,
                FRAME_SIZE,
                model_state,
                self.state.clone(),
                &self.config,
            );
            self.state = next;
            results.push(VadStreamFrameResult { probability, event });
            offset += FRAME_SIZE;
        }
        self.pending.drain(..offset);
        Ok(results)
    }

    /// Total samples consumed so far (excludes buffered partial frame).
    pub fn processed_samples(&self) -> usize {
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
}

/// The pure state machine, exposed for unit testing with synthetic probabilities.
/// Mirrors FluidAudio's `streamingStateMachine` exactly.
pub(crate) fn streaming_state_machine(
    probability: f32,
    chunk_sample_count: usize,
    model_state: ModelState,
    state: VadStreamState,
    config: &VadSegmentationConfig,
) -> (VadStreamState, Option<VadStreamEvent>) {
    let mut next = state;
    next.model_state = model_state;
    next.processed_samples += chunk_sample_count;

    let threshold = config.effective_threshold();
    let negative_threshold = config.effective_negative_threshold();
    let sr = SAMPLE_RATE as f64;
    let speech_pad_samples = (config.speech_padding * sr) as usize;
    let min_silence_samples = (config.min_silence_duration * sr) as usize;

    let mut event = None;

    if probability >= threshold {
        next.temp_end_sample = None;
        if !next.triggered {
            next.triggered = true;
            let raw_start = next.processed_samples as i64
                - speech_pad_samples as i64
                - chunk_sample_count as i64;
            event = Some(VadStreamEvent {
                kind: VadStreamEventKind::SpeechStart,
                sample_index: raw_start.max(0) as usize,
            });
        }
    } else if probability < negative_threshold && next.triggered {
        if next.temp_end_sample.is_none() {
            next.temp_end_sample = Some(next.processed_samples);
        }
        if let Some(silence_start) = next.temp_end_sample {
            if next.processed_samples - silence_start >= min_silence_samples {
                let raw_end =
                    silence_start as i64 + speech_pad_samples as i64 - chunk_sample_count as i64;
                next.triggered = false;
                next.temp_end_sample = None;
                event = Some(VadStreamEvent {
                    kind: VadStreamEventKind::SpeechEnd,
                    sample_index: raw_end.max(0) as usize,
                });
            }
        }
    }

    (next, event)
}
