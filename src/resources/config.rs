use bevy::post_process::bloom::Bloom;
use bevy::post_process::dof::DepthOfField;
use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;

use crate::resources::render::SimulationSettings;

#[derive(Resource, Clone, ExtractResource)]
pub struct PhysarumConfig {
    pub width: u32,
    pub height: u32,
    pub display_factor: u32,
    pub num_particles: u32,
    pub work_group_size: u32,
    pub decay_factor: f32,
    pub pixel_scale_factor: f32,
    pub deposit_factor: f32,
    pub color_mode: u32,
    pub action_area_size_sigma: f32,
    pub post_process_config: PostProcessConfig,
    pub liveliness_boost: f32,
    pub audio_reactive_enabled: bool,
    pub audio_input_mode: u32, // 0: microphone, 1: synthetic pulse
    pub audio_reactive_gain: f32,
    pub audio_bass_influence: f32,
    pub audio_mid_influence: f32,
    pub audio_treble_influence: f32,
    pub audio_beat_influence: f32,
    pub dark_profile_enabled: bool,
    pub dark_max_luminance: f32,
    pub dark_contrast: f32,
    pub dark_black_lift: f32,

    // Parameter selection/state (better placed in UI panel)
    pub settings_changed: bool,
    pub new_index: usize,
    pub current_settings: SimulationSettings,
}

impl Default for PhysarumConfig {
    fn default() -> Self {
        use crate::utils::load_parameters;

        let new_index = 0;
        let current_settings = load_parameters(new_index);

        Self {
            width: 1920,
            height: 1080,
            display_factor: 1,
            num_particles: 15_000_000,
            work_group_size: 32,
            decay_factor: 0.99,
            pixel_scale_factor: 1.0,
            deposit_factor: 1.0,
            color_mode: 12,
            action_area_size_sigma: 0.5,
            post_process_config: PostProcessConfig::default(),
            liveliness_boost: 0.65,
            audio_reactive_enabled: true,
            audio_input_mode: 0,
            audio_reactive_gain: 1.0,
            audio_bass_influence: 1.2,
            audio_mid_influence: 0.8,
            audio_treble_influence: 0.6,
            audio_beat_influence: 1.0,
            dark_profile_enabled: true,
            dark_max_luminance: 0.38,
            dark_contrast: 1.35,
            dark_black_lift: 0.01,
            settings_changed: false,
            new_index,
            current_settings,
        }
    }
}

#[derive(Clone)]
pub struct ChromaticAberrationConfig {
    pub intensity: f32,
}

impl Default for ChromaticAberrationConfig {
    fn default() -> Self {
        Self { intensity: 0.004 }
    }
}

#[derive(Clone)]
pub struct PostProcessConfig {
    pub bloom: Bloom,
    pub chromatic_aberration: ChromaticAberrationConfig,
    pub depth_of_field: DepthOfField,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        let bloom = Bloom {
            intensity: 0.04,
            ..Default::default()
        };
        let depth_of_field = DepthOfField {
            focal_distance: 35.0,
            max_depth: 70.0,
            ..Default::default()
        };

        Self {
            bloom,
            chromatic_aberration: ChromaticAberrationConfig::default(),
            depth_of_field,
        }
    }
}
