use bevy::prelude::Resource;

#[derive(Clone, Copy, Debug, Default)]
pub struct AudioFrame {
    pub level: f32,
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
}

#[derive(Resource, Default)]
pub struct AudioAnalysisState {
    pub mic_available: bool,
    pub level: f32,
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
    pub beat: f32,
    pub beat_hold: f32,
}

#[derive(Resource, Default)]
pub struct AudioCaptureStatus {
    pub mic_available: bool,
}
