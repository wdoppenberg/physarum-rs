use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;

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
}

impl Default for PhysarumConfig {
    fn default() -> Self {
        Self {
            width: 3024,
            height: 1964,
            display_factor: 1,
            num_particles: 25_000_000,
            work_group_size: 32,
            decay_factor: 0.99,
            pixel_scale_factor: 1.0,
            deposit_factor: 1.0,
            color_mode: 2,
        }
    }
}
