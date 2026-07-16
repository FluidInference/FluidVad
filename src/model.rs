//! Silero v6 inference via tract, model embedded in the binary.

use tract_nnef::prelude::*;

/// Sample rate the embedded model operates at.
pub const SAMPLE_RATE: usize = 16000;
/// New samples consumed per inference frame (32 ms at 16 kHz).
pub const FRAME_SIZE: usize = 512;
/// Samples of trailing context prepended to each frame (Silero convention).
pub const CONTEXT_SIZE: usize = 64;
/// LSTM hidden/cell width.
const STATE_SIZE: usize = 128;

/// The embedded model: upstream `silero_vad_16k_op15.onnx` with `sr` baked to
/// 16000 and every `If` node constant-folded away (`scripts/prepare_model.py`,
/// bit-exact with upstream under onnxruntime), then pre-compiled to NNEF
/// (`examples/export_nnef.rs`) so the runtime only needs the tract-nnef loader.
static MODEL_BYTES: &[u8] = include_bytes!("../models/silero_vad_16k_v6.nnef.tar");

/// Errors surfaced by FluidVad.
#[derive(Debug)]
pub enum FluidVadError {
    /// The embedded model failed to load or optimize (should never happen in a released build).
    ModelLoad(String),
    /// Inference failed.
    Inference(String),
    /// Caller passed a frame whose length is not [`FRAME_SIZE`].
    BadFrameSize { expected: usize, got: usize },
}

impl std::fmt::Display for FluidVadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelLoad(e) => write!(f, "model load failed: {e}"),
            Self::Inference(e) => write!(f, "inference failed: {e}"),
            Self::BadFrameSize { expected, got } => {
                write!(f, "frame must be exactly {expected} samples, got {got}")
            }
        }
    }
}

impl std::error::Error for FluidVadError {}

type Plan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Recurrent model state carried between frames: LSTM h/c plus the 64-sample
/// audio context prepended to the next frame.
#[derive(Clone, Debug)]
pub struct ModelState {
    pub(crate) hidden: Vec<f32>,
    pub(crate) cell: Vec<f32>,
    pub(crate) context: Vec<f32>,
}

impl Default for ModelState {
    fn default() -> Self {
        Self {
            hidden: vec![0.0; STATE_SIZE],
            cell: vec![0.0; STATE_SIZE],
            context: vec![0.0; CONTEXT_SIZE],
        }
    }
}

impl ModelState {
    /// Fresh all-zeros state, mirroring Silero's `reset_states`.
    pub fn initial() -> Self {
        Self::default()
    }
}

/// Silero VAD model. Construct once, reuse for every frame; cheap to share behind a reference.
pub struct SileroModel {
    plan: Plan,
}

impl SileroModel {
    /// Load the embedded pre-compiled graph. Takes a few ms; do it once.
    pub fn new() -> Result<Self, FluidVadError> {
        let plan = tract_nnef::nnef()
            .with_tract_core()
            .with_tract_resource()
            .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| FluidVadError::ModelLoad(e.to_string()))?;
        Ok(Self { plan })
    }

    /// Run one frame of exactly [`FRAME_SIZE`] samples (16 kHz mono, f32 in [-1, 1]).
    /// Returns the speech probability and the state to pass with the next frame.
    pub fn process_frame(
        &self,
        frame: &[f32],
        state: &ModelState,
    ) -> Result<(f32, ModelState), FluidVadError> {
        if frame.len() != FRAME_SIZE {
            return Err(FluidVadError::BadFrameSize {
                expected: FRAME_SIZE,
                got: frame.len(),
            });
        }

        let mut input = Vec::with_capacity(CONTEXT_SIZE + FRAME_SIZE);
        input.extend_from_slice(&state.context);
        input.extend_from_slice(frame);
        let input = tract_ndarray::Array2::from_shape_vec((1, CONTEXT_SIZE + FRAME_SIZE), input)
            .expect("shape is static")
            .into_tensor();

        let mut rnn = Vec::with_capacity(2 * STATE_SIZE);
        rnn.extend_from_slice(&state.hidden);
        rnn.extend_from_slice(&state.cell);
        let rnn = tract_ndarray::Array3::from_shape_vec((2, 1, STATE_SIZE), rnn)
            .expect("shape is static")
            .into_tensor();

        let out = self
            .plan
            .run(tvec![input.into(), rnn.into()])
            .map_err(|e| FluidVadError::Inference(e.to_string()))?;

        let prob = out[0]
            .to_array_view::<f32>()
            .map_err(|e| FluidVadError::Inference(e.to_string()))?[[0, 0]];
        let state_view = out[1]
            .to_array_view::<f32>()
            .map_err(|e| FluidVadError::Inference(e.to_string()))?;
        let flat: Vec<f32> = state_view.iter().copied().collect();

        Ok((
            prob,
            ModelState {
                hidden: flat[..STATE_SIZE].to_vec(),
                cell: flat[STATE_SIZE..].to_vec(),
                context: frame[FRAME_SIZE - CONTEXT_SIZE..].to_vec(),
            },
        ))
    }

    /// Speech probability for every full frame in `samples` (tail shorter than a frame is dropped).
    pub fn probabilities(&self, samples: &[f32]) -> Result<Vec<f32>, FluidVadError> {
        let mut state = ModelState::initial();
        let mut probs = Vec::with_capacity(samples.len() / FRAME_SIZE);
        for frame in samples.chunks_exact(FRAME_SIZE) {
            let (p, next) = self.process_frame(frame, &state)?;
            probs.push(p);
            state = next;
        }
        Ok(probs)
    }
}
