use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;

// Main world resource to track input state
#[derive(Resource, Clone, ExtractResource)]
pub struct PhysarumInputState {
    pub settings_changed: bool,
    pub new_index: usize,
}

impl Default for PhysarumInputState {
    fn default() -> Self {
        Self {
            settings_changed: false,
            new_index: 0,
        }
    }
}
