use crate::resources::audio::{AudioAnalysisState, AudioCaptureStatus, AudioFrame};
use crate::resources::config::PhysarumConfig;
use crate::resources::input::PhysarumInputState;
use bevy::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{mpsc, Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
pub struct AudioCaptureThread {
    pub rx: Arc<Mutex<mpsc::Receiver<AudioFrame>>>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct FilterState {
    low_env: f32,
    mid_env: f32,
    high_env: f32,
}

#[cfg(not(target_arch = "wasm32"))]
fn alpha_for_cutoff(cutoff_hz: f32, sample_rate_hz: f32) -> f32 {
    let x = (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate_hz).exp();
    (1.0 - x).clamp(0.0001, 0.99)
}

#[cfg(not(target_arch = "wasm32"))]
fn analyze_chunk(
    data: &[f32],
    channels: usize,
    sample_rate_hz: f32,
    state: &mut FilterState,
) -> AudioFrame {
    let alpha_low = alpha_for_cutoff(180.0, sample_rate_hz);
    let alpha_mid = alpha_for_cutoff(1200.0, sample_rate_hz);
    let alpha_high = alpha_for_cutoff(3200.0, sample_rate_hz);

    let mut level_sum = 0.0;
    let mut bass_sum = 0.0;
    let mut mid_sum = 0.0;
    let mut treble_sum = 0.0;
    let mut count = 0.0;

    for frame in data.chunks(channels.max(1)) {
        let mut mono = 0.0;
        for sample in frame {
            mono += *sample;
        }
        mono /= frame.len() as f32;

        let x = mono.abs();
        state.low_env += alpha_low * (x - state.low_env);
        state.mid_env += alpha_mid * (x - state.mid_env);
        state.high_env += alpha_high * (x - state.high_env);

        let bass = state.low_env;
        let mid = (state.mid_env - state.low_env).abs();
        let treble = (x - state.high_env).abs();

        level_sum += x;
        bass_sum += bass;
        mid_sum += mid;
        treble_sum += treble;
        count += 1.0;
    }

    if count < 1.0 {
        return AudioFrame::default();
    }

    let level = (level_sum / count * 3.0).clamp(0.0, 1.0);
    let bass = (bass_sum / count * 6.0).clamp(0.0, 1.0);
    let mid = (mid_sum / count * 10.0).clamp(0.0, 1.0);
    let treble = (treble_sum / count * 18.0).clamp(0.0, 1.0);

    AudioFrame {
        level,
        bass,
        mid,
        treble,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn build_stream_for_format(
    device: &cpal::Device,
    stream_config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    sender: mpsc::Sender<AudioFrame>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    let sample_rate_hz = stream_config.sample_rate.0 as f32;
    let channels = stream_config.channels as usize;
    let mut filter_state = FilterState::default();

    let err_fn = |err| warn!("Audio input stream error: {err:?}");

    match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            stream_config,
            move |data: &[f32], _| {
                let frame = analyze_chunk(data, channels, sample_rate_hz, &mut filter_state);
                let _ = sender.send(frame);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            stream_config,
            move |data: &[i16], _| {
                let mut scratch = Vec::with_capacity(data.len());
                for sample in data {
                    scratch.push(*sample as f32 / i16::MAX as f32);
                }
                let frame = analyze_chunk(&scratch, channels, sample_rate_hz, &mut filter_state);
                let _ = sender.send(frame);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            stream_config,
            move |data: &[u16], _| {
                let mut scratch = Vec::with_capacity(data.len());
                for sample in data {
                    scratch.push((*sample as f32 / u16::MAX as f32) * 2.0 - 1.0);
                }
                let frame = analyze_chunk(&scratch, channels, sample_rate_hz, &mut filter_state);
                let _ = sender.send(frame);
            },
            err_fn,
            None,
        ),
        _ => unreachable!("Unsupported CPAL sample format"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_capture_thread(sender: mpsc::Sender<AudioFrame>) {
    let host = cpal::default_host();
    let Some(device) = host.default_input_device() else {
        warn!("No input audio device found; synthetic audio mode remains available");
        return;
    };

    let Ok(supported_config) = device.default_input_config() else {
        warn!("No default input config found; synthetic audio mode remains available");
        return;
    };

    let stream_config: cpal::StreamConfig = supported_config.clone().into();
    let Ok(stream) = build_stream_for_format(
        &device,
        &stream_config,
        supported_config.sample_format(),
        sender,
    ) else {
        warn!("Failed to create microphone stream; synthetic mode remains available");
        return;
    };

    if let Err(err) = stream.play() {
        warn!("Failed to start microphone stream: {err:?}");
        return;
    }

    info!("Microphone audio stream started for reactive controls");

    // Keep the stream alive in this detached thread.
    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn setup_audio_capture(mut commands: Commands) {
    let (tx, rx) = mpsc::channel::<AudioFrame>();
    std::thread::Builder::new()
        .name("physarum-audio-capture".to_string())
        .spawn(move || run_capture_thread(tx))
        .ok();

    commands.insert_resource(AudioCaptureStatus {
        mic_available: false,
    });
    commands.insert_resource(AudioCaptureThread {
        rx: Arc::new(Mutex::new(rx)),
    });
    commands.init_resource::<AudioAnalysisState>();
}

#[cfg(target_arch = "wasm32")]
pub fn setup_audio_capture(mut commands: Commands) {
    commands.insert_resource(AudioCaptureStatus {
        mic_available: false,
    });
    commands.init_resource::<AudioAnalysisState>();
}

fn smooth_towards(current: f32, target: f32, alpha: f32) -> f32 {
    current + (target - current) * alpha
}

pub fn update_audio_reactivity(
    time: Res<Time>,
    config: Res<PhysarumConfig>,
    mut capture_status: ResMut<AudioCaptureStatus>,
    mut analysis: ResMut<AudioAnalysisState>,
    mut input_state: ResMut<PhysarumInputState>,
    #[cfg(not(target_arch = "wasm32"))] capture: Option<Res<AudioCaptureThread>>,
) {
    analysis.mic_available = capture_status.mic_available;

    let mut target = AudioFrame::default();
    let use_mic = config.audio_reactive_enabled && config.audio_input_mode == 0;
    let mut got_mic_frame = false;

    if use_mic {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(capture) = capture {
            if let Ok(rx) = capture.rx.lock() {
                while let Ok(frame) = rx.try_recv() {
                    target = frame;
                    capture_status.mic_available = true;
                    got_mic_frame = true;
                }
            }
        }
    }

    if config.audio_reactive_enabled && (!use_mic || !got_mic_frame) {
        let t = time.elapsed_secs();
        let bass = (0.5 + 0.5 * (t * 2.1).sin()).powf(1.6);
        let mid = (0.5 + 0.5 * (t * 3.9 + 0.8).sin()).powf(1.8);
        let treble = (0.5 + 0.5 * (t * 8.2 + 2.4).sin()).powf(2.2);
        let beat = (0.5 + 0.5 * (t * 2.0 * std::f32::consts::PI * 1.95).sin()).powf(8.0);
        target = AudioFrame {
            level: (0.35 * bass + 0.4 * mid + 0.25 * treble).clamp(0.0, 1.0),
            bass: (bass + beat * 0.6).clamp(0.0, 1.0),
            mid,
            treble,
        };
    }

    analysis.level = smooth_towards(analysis.level, target.level, 0.2);
    analysis.bass = smooth_towards(analysis.bass, target.bass, 0.18);
    analysis.mid = smooth_towards(analysis.mid, target.mid, 0.2);
    analysis.treble = smooth_towards(analysis.treble, target.treble, 0.25);

    let beat_energy =
        (analysis.bass * 0.65 + analysis.mid * 0.25 + analysis.treble * 0.10).clamp(0.0, 1.0);
    analysis.beat_hold = (analysis.beat_hold - time.delta_secs()).max(0.0);
    if beat_energy > 0.72 && analysis.beat_hold <= 0.0 {
        analysis.beat = 1.0;
        analysis.beat_hold = 0.12;
    } else {
        analysis.beat = (analysis.beat - time.delta_secs() * 5.5).max(0.0);
    }

    if !config.audio_reactive_enabled {
        input_state.audio_level = 0.0;
        input_state.audio_bass = 0.0;
        input_state.audio_mid = 0.0;
        input_state.audio_treble = 0.0;
        input_state.audio_beat = 0.0;
        return;
    }

    let gain = config.audio_reactive_gain.max(0.0);
    input_state.audio_level = (analysis.level * gain).clamp(0.0, 1.0);
    input_state.audio_bass = (analysis.bass * gain).clamp(0.0, 1.0);
    input_state.audio_mid = (analysis.mid * gain).clamp(0.0, 1.0);
    input_state.audio_treble = (analysis.treble * gain).clamp(0.0, 1.0);
    input_state.audio_beat = (analysis.beat * gain).clamp(0.0, 1.0);
}
