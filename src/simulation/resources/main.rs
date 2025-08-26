use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;

use crate::simulation::utils::load_parameters;
use crate::simulation::resources::render::PointSettings;

// Main world resource to track input state
#[derive(Resource, Clone, ExtractResource)]
pub struct PhysarumInputState {
    pub settings_changed: bool,
    pub new_index: usize,
    pub current_settings: PointSettings,
}

impl Default for PhysarumInputState {
    fn default() -> Self {
        let new_index = 0;
        let current_settings = load_parameters(new_index);
        Self {
            settings_changed: false,
            new_index,
            current_settings,
        }
    }
}
