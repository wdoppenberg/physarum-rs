use bevy::render::render_resource::ShaderType;

#[derive(ShaderType)]
pub(crate) struct UniformData {
    pub(crate) width: u32,
    pub(crate) height: u32,
    // A generic value field used by different passes:
    // - setter: reset value
    // - deposit: depositFactor
    // - move: pixelScaleFactor
    // - diffusion: decayFactor
    pub(crate) value: f32,
    pub(crate) color_mode: u32,
    // Extended fields used by move shader (others may ignore):
    pub(crate) num_particles: u32,
    pub(crate) time: f32,
    pub(crate) action_area_size_sigma: f32,
    pub(crate) action_x: f32,
    pub(crate) action_y: f32,
    pub(crate) move_bias_action_x: f32,
    pub(crate) move_bias_action_y: f32,
    pub(crate) l2_action: f32,
    pub(crate) spawn_particles: u32,
    pub(crate) spawn_fraction: f32,
    pub(crate) random_spawn_number: u32,
}
