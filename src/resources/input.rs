use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;

// Main world resource to track input state and interactive uniforms
#[derive(Resource, Clone, ExtractResource)]
pub struct PhysarumInputState {
    // Interactive uniforms
    pub time: f32,
    pub action_x: f32,
    pub action_y: f32,
    pub move_bias_action_x: f32,
    pub move_bias_action_y: f32,
    pub l2_action: f32,
    pub spawn_particles: u32,
    pub spawn_fraction: f32,
    pub random_spawn_number: u32,
    pub num_boids: u32,
    pub audio_level: f32,
    pub audio_bass: f32,
    pub audio_mid: f32,
    pub audio_treble: f32,
    pub audio_beat: f32,
}

impl Default for PhysarumInputState {
    fn default() -> Self {
        Self {
            time: 0.0,
            action_x: 0.0,
            action_y: 0.0,
            move_bias_action_x: 0.0,
            move_bias_action_y: 0.0,
            l2_action: 0.0,
            spawn_particles: 0,
            spawn_fraction: 0.15,
            random_spawn_number: 0,
            num_boids: 0,
            audio_level: 0.0,
            audio_bass: 0.0,
            audio_mid: 0.0,
            audio_treble: 0.0,
            audio_beat: 0.0,
        }
    }
}
