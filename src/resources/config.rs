use bevy::post_process::bloom::Bloom;
use bevy::post_process::dof::DepthOfField;
use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;

use crate::resources::render::PointSettings;

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

    // Parameter selection/state (better placed in UI panel)
    pub settings_changed: bool,
    pub new_index: usize,
    pub current_settings: PointSettings,
}

impl Default for PhysarumConfig {
    fn default() -> Self {
        use crate::utils::load_parameters;

        let new_index = 0;
        let current_settings = load_parameters(new_index);

        Self {
            width: 3024,
            height: 1964,
            display_factor: 1,
            num_particles: 5_000_000,
            work_group_size: 32,
            decay_factor: 0.99,
            pixel_scale_factor: 1.0,
            deposit_factor: 1.0,
            color_mode: 0,
            action_area_size_sigma: 0.2,
            post_process_config: PostProcessConfig::default(),
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
        Self { intensity: 0.01 }
    }
}

#[derive(Default, Clone)]
pub struct PostProcessConfig {
    pub bloom: Bloom,
    pub chromatic_aberration: ChromaticAberrationConfig,
    pub depth_of_field: DepthOfField
}
