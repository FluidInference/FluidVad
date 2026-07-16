//! Streaming wrapper tests: framing, event emission on real audio.

use fluidvad::{SileroModel, VadSegmentationConfig, VadStreamer, FRAME_SIZE};

fn load_test_wav() -> Option<Vec<f32>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/speech_16k.wav");
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000);
    assert_eq!(spec.channels, 1);
    Some(
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
    )
}

#[test]
fn streamer_buffers_arbitrary_chunk_sizes() {
    let Some(samples) = load_test_wav() else {
        eprintln!("SKIP: testdata/speech_16k.wav not present");
        return;
    };
    let model = SileroModel::new().expect("model");
    let mut streamer = VadStreamer::with_model(model, VadSegmentationConfig::default());

    // push in awkward chunk sizes; frame count must equal len / FRAME_SIZE
    let mut results = Vec::new();
    for chunk in samples.chunks(1000) {
        results.extend(streamer.push(chunk).expect("push"));
    }
    assert_eq!(results.len(), samples.len() / FRAME_SIZE);
    assert_eq!(streamer.processed_samples(), results.len() * FRAME_SIZE);
}

#[test]
fn streamer_emits_start_and_end_on_real_speech() {
    let Some(mut samples) = load_test_wav() else {
        eprintln!("SKIP: testdata/speech_16k.wav not present");
        return;
    };
    // append 1.5s of silence so the final speech run closes
    samples.extend(std::iter::repeat_n(0.0f32, 24000));

    let model = SileroModel::new().expect("model");
    let mut streamer = VadStreamer::with_model(model, VadSegmentationConfig::default());
    let results = streamer.push(&samples).expect("push");

    let starts: Vec<_> = results
        .iter()
        .filter_map(|r| r.event.filter(|e| e.is_start()))
        .collect();
    let ends: Vec<_> = results
        .iter()
        .filter_map(|r| r.event.filter(|e| e.is_end()))
        .collect();
    assert!(!starts.is_empty(), "no SpeechStart on real speech");
    assert!(!ends.is_empty(), "no SpeechEnd after trailing silence");
    assert!(starts[0].sample_index < ends.last().unwrap().sample_index);
    // events alternate: start, end, start, end...
    let mut expect_start = true;
    for r in results.iter().filter_map(|r| r.event) {
        assert_eq!(r.is_start(), expect_start, "events must alternate");
        expect_start = !expect_start;
    }
    assert!(
        !streamer.is_speaking(),
        "trailing silence must close speech"
    );
}
