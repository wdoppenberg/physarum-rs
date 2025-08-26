use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;

use crate::simulation::utils::load_parameters;
use crate::simulation::resources::render::PointSettings;

// Main world resource to track input state and interactive uniforms
#[derive(Resource, Clone, ExtractResource)]
pub struct PhysarumInputState {
    // Parameter selection/state
    pub settings_changed: bool,
    pub new_index: usize,
    pub current_settings: PointSettings,

    // Interactive uniforms
    pub time: f32,
    pub action_area_size_sigma: f32,
    pub action_x: f32,
    pub action_y: f32,
    pub move_bias_action_x: f32,
    pub move_bias_action_y: f32,
    pub l2_action: f32,
    pub spawn_particles: u32,
    pub spawn_fraction: f32,
    pub random_spawn_number: u32,
    pub color_mode: u32,
}

impl Default for PhysarumInputState {
    fn default() -> Self {
        let new_index = 0;
        let current_settings = load_parameters(new_index);
        Self {
            settings_changed: false,
            new_index,
            current_settings,
            time: 0.0,
            action_area_size_sigma: 0.2,
            action_x: 0.0,
            action_y: 0.0,
            move_bias_action_x: 0.0,
            move_bias_action_y: 0.0,
            l2_action: 0.0,
            spawn_particles: 0,
            spawn_fraction: 0.15,
            random_spawn_number: 0,
            color_mode: crate::simulation::constants::COLOR_MODE,
        }
    }
}
