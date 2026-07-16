//! Replicates FluidAudio's `vad-benchmark` protocol for direct comparison:
//! per-chunk probability >= threshold => chunk active; a clip is predicted
//! speech when >= `activity` fraction of its chunks are active. Metrics are
//! accuracy / precision / recall / F1 over labeled speech + noise clips.
//!
//! FluidAudio's CoreML model emits one probability per 4096-sample chunk
//! (256 ms); FluidVad emits one per 512-sample frame (32 ms). `--agg chunks`
//! max-pools 8 frames to match the 256 ms granularity; `--agg frames` uses
//! raw frames.
//!
//! Usage:
//!   cargo run --release --example vad_benchmark -- \
//!     --speech <dir> --noise <dir> [--threshold 0.3] [--activity 0.1] [--agg chunks]

use fluidvad::SileroModel;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn wav_samples(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000, "{path:?} must be 16 kHz");
    assert_eq!(spec.channels, 1, "{path:?} must be mono");
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn wavs_in(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir:?}: {e}")) {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|x| x == "wav") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(dir, &mut files);
    files.sort();
    files
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let speech_dir = PathBuf::from(get("--speech").expect("--speech <dir> required"));
    let noise_dir = PathBuf::from(get("--noise").expect("--noise <dir> required"));
    let threshold: f32 = get("--threshold")
        .map(|v| v.parse().unwrap())
        .unwrap_or(0.3);
    let activity: f32 = get("--activity").map(|v| v.parse().unwrap()).unwrap_or(0.1);
    let agg = get("--agg").unwrap_or_else(|| "chunks".into());
    let max_files: usize = get("--max-files")
        .map(|v| v.parse().unwrap())
        .unwrap_or(usize::MAX);

    let model = SileroModel::new().expect("model");

    // (path, label): 1 = speech, 0 = noise — same labeling as FluidAudio
    let mut files: Vec<(PathBuf, u8)> = Vec::new();
    files.extend(
        wavs_in(&speech_dir)
            .into_iter()
            .take(max_files)
            .map(|p| (p, 1u8)),
    );
    files.extend(
        wavs_in(&noise_dir)
            .into_iter()
            .take(max_files)
            .map(|p| (p, 0u8)),
    );

    let (mut tp, mut fp, mut tn, mut fn_) = (0u32, 0u32, 0u32, 0u32);
    let mut total_audio = 0.0f64;
    let mut total_infer = 0.0f64;
    let mut misclassified: Vec<String> = Vec::new();

    for (path, label) in &files {
        let samples = wav_samples(path);
        total_audio += samples.len() as f64 / 16000.0;

        let t = Instant::now();
        let frame_probs = model.probabilities(&samples).expect("probabilities");
        total_infer += t.elapsed().as_secs_f64();

        // aggregate to FluidAudio's 256 ms chunk granularity (max over 8 frames)
        let probs: Vec<f32> = if agg == "chunks" {
            frame_probs
                .chunks(8)
                .map(|c| c.iter().cloned().fold(0.0f32, f32::max))
                .collect()
        } else {
            frame_probs
        };

        let active = probs.iter().filter(|&&p| p >= threshold).count();
        let ratio = active as f32 / probs.len().max(1) as f32;
        let predicted: u8 = if ratio >= activity { 1 } else { 0 };

        match (label, predicted) {
            (1, 1) => tp += 1,
            (1, 0) => fn_ += 1,
            (0, 1) => fp += 1,
            (0, 0) => tn += 1,
            _ => unreachable!(),
        }
        if *label != predicted {
            misclassified.push(format!(
                "{} (label {label}, ratio {ratio:.3})",
                path.file_name().unwrap().to_string_lossy()
            ));
        }
    }

    let n = (tp + fp + tn + fn_) as f32;
    let accuracy = (tp + tn) as f32 / n;
    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f32 / (tp + fn_) as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!(
        "files: {} ({} speech / {} noise)",
        files.len(),
        tp + fn_,
        tn + fp
    );
    println!("threshold: {threshold}  activity: {activity}  agg: {agg}");
    println!("accuracy:  {:.1}%", accuracy * 100.0);
    println!("precision: {:.1}%", precision * 100.0);
    println!("recall:    {:.1}%", recall * 100.0);
    println!("f1:        {:.1}%", f1 * 100.0);
    println!(
        "rtfx:      {:.0}x ({:.1}s audio in {:.2}s)",
        total_audio / total_infer,
        total_audio,
        total_infer
    );
    if !misclassified.is_empty() {
        println!("misclassified ({}):", misclassified.len());
        for m in &misclassified {
            println!("  {m}");
        }
    }
}
