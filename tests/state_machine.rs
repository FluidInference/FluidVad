//! State-machine unit tests with synthetic probability sequences.
//! Mirrors FluidAudio's approach of exposing the pure machine for testing —
//! synthetic *probabilities* exercise the logic, no model involved.

use fluidvad::{segment_from_probabilities, VadSegmentationConfig, FRAME_SIZE, SAMPLE_RATE};

fn config() -> VadSegmentationConfig {
    VadSegmentationConfig::default()
}

/// frames -> samples
fn s(frames: usize) -> usize {
    frames * FRAME_SIZE
}

#[test]
fn empty_input_yields_no_segments() {
    assert!(segment_from_probabilities(&[], 0, &config()).is_empty());
    assert!(segment_from_probabilities(&[0.9], 0, &config()).is_empty());
}

#[test]
fn all_silence_yields_no_segments() {
    let probs = vec![0.05f32; 100];
    assert!(segment_from_probabilities(&probs, s(100), &config()).is_empty());
}

#[test]
fn basic_speech_run_is_detected_with_padding() {
    // 10 silence, 40 speech, 40 silence (min_silence 0.75s = 23.4 frames)
    let mut probs = vec![0.05f32; 10];
    probs.extend(vec![0.95f32; 40]);
    probs.extend(vec![0.05f32; 40]);
    let total = s(probs.len());
    let segs = segment_from_probabilities(&probs, total, &config());
    assert_eq!(segs.len(), 1);
    let pad = 0.1;
    let expected_start = (s(10) as f64 / SAMPLE_RATE as f64) - pad;
    let expected_end = (s(50) as f64 / SAMPLE_RATE as f64) + pad;
    assert!(
        (segs[0].start_time - expected_start).abs() < 1e-9,
        "{:?}",
        segs[0]
    );
    assert!(
        (segs[0].end_time - expected_end).abs() < 1e-9,
        "{:?}",
        segs[0]
    );
}

#[test]
fn short_blip_below_min_speech_is_dropped() {
    // 0.15s min speech = 4.7 frames; a 3-frame blip must be dropped
    let mut probs = vec![0.05f32; 10];
    probs.extend(vec![0.95f32; 3]);
    probs.extend(vec![0.05f32; 40]);
    let segs = segment_from_probabilities(&probs, s(probs.len()), &config());
    assert!(segs.is_empty(), "{segs:?}");
}

#[test]
fn brief_dip_does_not_split_segment() {
    // dip of 5 frames (0.16s) < min_silence (0.75s): one segment
    let mut probs = vec![0.95f32; 30];
    probs.extend(vec![0.05f32; 5]);
    probs.extend(vec![0.95f32; 30]);
    probs.extend(vec![0.05f32; 40]);
    let segs = segment_from_probabilities(&probs, s(probs.len()), &config());
    assert_eq!(segs.len(), 1, "{segs:?}");
}

#[test]
fn long_silence_splits_segments() {
    // 30 frames silence (0.96s) > min_silence: two segments
    let mut probs = vec![0.95f32; 30];
    probs.extend(vec![0.05f32; 30]);
    probs.extend(vec![0.95f32; 30]);
    probs.extend(vec![0.05f32; 40]);
    let segs = segment_from_probabilities(&probs, s(probs.len()), &config());
    assert_eq!(segs.len(), 2, "{segs:?}");
    assert!(segs[0].end_time < segs[1].start_time);
}

#[test]
fn hysteresis_keeps_segment_open_between_thresholds() {
    // probs between negative (0.35) and entry (0.5) must not close the segment
    let mut probs = vec![0.95f32; 20];
    probs.extend(vec![0.40f32; 30]); // in the hysteresis band
    probs.extend(vec![0.95f32; 20]);
    probs.extend(vec![0.05f32; 40]);
    let segs = segment_from_probabilities(&probs, s(probs.len()), &config());
    assert_eq!(segs.len(), 1, "{segs:?}");
}

#[test]
fn max_speech_duration_forces_split_at_candidate_silence() {
    // continuous speech far beyond 14s with a mid dip below silence_threshold_for_split
    let mut probs = vec![0.95f32; 220]; // ~7s
    probs.extend(vec![0.10f32; 4]); // candidate silence (0.128s > minSilenceAtMaxSpeech 0.098s), below 0.3
    probs.extend(vec![0.95f32; 300]); // ~9.6s, total > 14s
    probs.extend(vec![0.05f32; 40]);
    let segs = segment_from_probabilities(&probs, s(probs.len()), &config());
    assert!(segs.len() >= 2, "expected forced split, got {segs:?}");
    for seg in &segs {
        assert!(seg.duration() <= 14.0 + 0.3, "segment too long: {seg:?}");
    }
}

#[test]
fn close_segments_share_padding_without_overlap() {
    // Force a split with a gap (5 frames = 0.16s) smaller than 2*padding (0.2s)
    // by lowering min_silence: the padding pass must split the gap in half
    // rather than overlap the segments.
    let cfg = VadSegmentationConfig {
        min_silence_duration: 0.1,
        ..VadSegmentationConfig::default()
    };
    let mut probs = vec![0.95f32; 30];
    probs.extend(vec![0.05f32; 5]);
    probs.extend(vec![0.95f32; 30]);
    probs.extend(vec![0.05f32; 40]);
    let segs = segment_from_probabilities(&probs, s(probs.len()), &cfg);
    assert_eq!(segs.len(), 2, "{segs:?}");
    assert!(
        segs[0].end_time <= segs[1].start_time,
        "padding must not overlap: {segs:?}"
    );
    // gap was split in half: boundaries meet in the middle of the silence
    let gap_mid = (s(30) + s(35)) as f64 / 2.0 / SAMPLE_RATE as f64;
    assert!(
        (segs[0].end_time - gap_mid).abs() < 0.02,
        "{segs:?} mid={gap_mid}"
    );
}

#[test]
fn pinned_negative_threshold_derives_entry_threshold() {
    let cfg = VadSegmentationConfig {
        negative_threshold: Some(0.2),
        ..VadSegmentationConfig::default()
    };
    // entry = 0.2 + 0.15 = 0.35: probs of 0.4 must now trigger
    let mut probs = vec![0.40f32; 30];
    probs.extend(vec![0.05f32; 40]);
    let segs = segment_from_probabilities(&probs, s(probs.len()), &cfg);
    assert_eq!(segs.len(), 1, "{segs:?}");
}
