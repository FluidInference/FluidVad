//! wasm-bindgen bindings — the `@fluidinference/fluidvad` npm API surface.

use wasm_bindgen::prelude::*;

use crate::model::FRAME_SIZE;
use crate::segmentation::{segment_speech, VadSegmentationConfig};
use crate::streaming::VadStreamer;

/// Configuration options; all fields optional on the JS side.
#[wasm_bindgen]
#[derive(Clone, Copy, Default)]
pub struct VadOptions {
    threshold: Option<f32>,
    min_speech_duration: Option<f64>,
    min_silence_duration: Option<f64>,
    max_speech_duration: Option<f64>,
    speech_padding: Option<f64>,
    negative_threshold: Option<f32>,
}

#[wasm_bindgen]
impl VadOptions {
    #[wasm_bindgen(constructor)]
    pub fn new() -> VadOptions {
        VadOptions::default()
    }

    #[wasm_bindgen(setter)]
    pub fn set_threshold(&mut self, v: f32) {
        self.threshold = Some(v);
    }

    #[wasm_bindgen(setter, js_name = minSpeechDuration)]
    pub fn set_min_speech_duration(&mut self, v: f64) {
        self.min_speech_duration = Some(v);
    }

    #[wasm_bindgen(setter, js_name = minSilenceDuration)]
    pub fn set_min_silence_duration(&mut self, v: f64) {
        self.min_silence_duration = Some(v);
    }

    #[wasm_bindgen(setter, js_name = maxSpeechDuration)]
    pub fn set_max_speech_duration(&mut self, v: f64) {
        self.max_speech_duration = Some(v);
    }

    #[wasm_bindgen(setter, js_name = speechPadding)]
    pub fn set_speech_padding(&mut self, v: f64) {
        self.speech_padding = Some(v);
    }

    #[wasm_bindgen(setter, js_name = negativeThreshold)]
    pub fn set_negative_threshold(&mut self, v: f32) {
        self.negative_threshold = Some(v);
    }
}

impl VadOptions {
    fn to_config(self) -> VadSegmentationConfig {
        let mut cfg = VadSegmentationConfig::default();
        if let Some(v) = self.threshold {
            cfg.threshold = v;
        }
        if let Some(v) = self.min_speech_duration {
            cfg.min_speech_duration = v;
        }
        if let Some(v) = self.min_silence_duration {
            cfg.min_silence_duration = v;
        }
        if let Some(v) = self.max_speech_duration {
            cfg.max_speech_duration = v;
        }
        if let Some(v) = self.speech_padding {
            cfg.speech_padding = v;
        }
        cfg.negative_threshold = self.negative_threshold.or(cfg.negative_threshold);
        cfg
    }
}

/// A speech boundary event.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct VadEvent {
    /// `true` for SpeechStart, `false` for SpeechEnd.
    #[wasm_bindgen(js_name = isStart, readonly)]
    pub is_start: bool,
    /// Boundary position in samples (16 kHz, padding applied).
    #[wasm_bindgen(js_name = sampleIndex, readonly)]
    pub sample_index: usize,
    /// Boundary position in seconds.
    #[wasm_bindgen(js_name = timeSeconds, readonly)]
    pub time_seconds: f64,
}

/// A detected speech region, seconds.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Segment {
    #[wasm_bindgen(js_name = startTime, readonly)]
    pub start_time: f64,
    #[wasm_bindgen(js_name = endTime, readonly)]
    pub end_time: f64,
}

/// Silero VAD: streaming events + offline segmentation. Model is embedded; no fetches.
#[wasm_bindgen]
pub struct FluidVad {
    streamer: VadStreamer,
    last_probability: f32,
}

#[wasm_bindgen]
impl FluidVad {
    /// Build the VAD (parses + optimizes the embedded model; do this once).
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<VadOptions>) -> Result<FluidVad, JsError> {
        let config = options.unwrap_or_default().to_config();
        let streamer = VadStreamer::new(config).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(FluidVad {
            streamer,
            last_probability: 0.0,
        })
    }

    /// Samples per frame (512 = 32 ms at 16 kHz).
    #[wasm_bindgen(js_name = frameSize)]
    pub fn frame_size() -> usize {
        FRAME_SIZE
    }

    /// Push 16 kHz mono f32 samples of any length; returns boundary events for
    /// every completed frame. Partial frames are buffered internally.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<VadEvent>, JsError> {
        let results = self
            .streamer
            .push(samples)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let mut events = Vec::new();
        for r in results {
            self.last_probability = r.probability;
            if let Some(e) = r.event {
                events.push(VadEvent {
                    is_start: e.is_start(),
                    sample_index: e.sample_index,
                    time_seconds: e.time_seconds(),
                });
            }
        }
        Ok(events)
    }

    /// Offline: segment a whole buffer into speech regions.
    pub fn segment(&self, samples: &[f32]) -> Result<Vec<Segment>, JsError> {
        let config = VadSegmentationConfig::default();
        segment_speech(self.streamer.model(), samples, &config)
            .map(|segs| {
                segs.into_iter()
                    .map(|s| Segment {
                        start_time: s.start_time,
                        end_time: s.end_time,
                    })
                    .collect()
            })
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Speech probability of the most recently processed frame.
    #[wasm_bindgen(getter, js_name = lastProbability)]
    pub fn last_probability(&self) -> f32 {
        self.last_probability
    }

    /// Whether the state machine is currently inside speech.
    #[wasm_bindgen(getter, js_name = isSpeaking)]
    pub fn is_speaking(&self) -> bool {
        self.streamer.is_speaking()
    }

    /// Reset model + hysteresis state and drop buffered samples.
    pub fn reset(&mut self) {
        self.streamer.reset();
    }
}
