//! FluidVad — Silero VAD in pure Rust.
//!
//! CPU-only ONNX inference via [tract](https://github.com/sonos/tract), with the
//! Silero v6 16 kHz model embedded in the binary. One core, two targets:
//! native (crates.io) and wasm32 (npm).

mod model;
mod segmentation;
mod streaming;

pub use model::{FluidVadError, ModelState, SileroModel, CONTEXT_SIZE, FRAME_SIZE, SAMPLE_RATE};
pub use segmentation::{
    segment_from_probabilities, segment_speech, VadSegment, VadSegmentationConfig,
};
pub use streaming::{
    VadStreamEvent, VadStreamEventKind, VadStreamFrameResult, VadStreamState, VadStreamer,
};

#[cfg(target_arch = "wasm32")]
mod wasm;
