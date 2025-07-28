use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;

// Main world resource to track input state
#[derive(Default, Resource, Clone, ExtractResource)]
pub struct PhysarumInputState {
    pub settings_changed: bool,
    pub new_index: usize,
}
