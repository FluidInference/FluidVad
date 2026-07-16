//! Regenerate the embedded NNEF artifact from the prepared ONNX model.
//!
//! ONNX parsing + graph optimization happen here, at build time; the shipped
//! library only carries the lightweight tract-nnef loader.
//!
//! Usage: cargo run --release --example export_nnef

use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        .model_for_path("models/silero_vad_16k_v6.onnx")?
        .with_input_fact(0, f32::fact([1, 576]).into())?
        .with_input_fact(1, f32::fact([2, 1, 128]).into())?
        .into_typed()?
        .into_decluttered()?;

    let nnef = tract_nnef::nnef().with_tract_core().with_tract_resource();
    let mut tar = Vec::new();
    nnef.write_to_tar(&model, &mut tar)?;
    std::fs::write("models/silero_vad_16k_v6.nnef.tar", &tar)?;
    println!(
        "wrote models/silero_vad_16k_v6.nnef.tar ({} bytes)",
        tar.len()
    );
    Ok(())
}
