use bevy::prelude::*;
use bevy::render::render_resource::*;

/// Settings for the simulation points
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PointSettings {
    pub default_scaling_factor: f32,
    pub sensor_distance0: f32,
    pub sd_exponent: f32,
    pub sd_amplitude: f32,
    pub sensor_angle0: f32,
    pub sa_exponent: f32,
    pub sa_amplitude: f32,
    pub rotation_angle0: f32,
    pub ra_exponent: f32,
    pub ra_amplitude: f32,
    pub move_distance0: f32,
    pub md_exponent: f32,
    pub md_amplitude: f32,
    pub sensor_bias1: f32,
    pub sensor_bias2: f32,
}

/// Main resource for the physarum simulation
#[derive(Resource)]
pub struct PhysarumSimulation {
    /// Index of the current point settings in the parameters matrix
    pub point_cursor_index: usize,
    
    /// Buffer containing the simulation parameters
    pub simulation_params_buffer: Buffer,
    
    /// Buffer containing uniform values (width, height, etc.)
    pub uniform_buffer: Buffer,
    
    /// Buffer for counting particles in each cell
    pub counter_buffer: Buffer,
    
    /// Ping-pong textures for the trail
    pub trail_texture_a: Texture,
    pub trail_texture_view_a: TextureView,
    pub trail_texture_b: Texture,
    pub trail_texture_view_b: TextureView,
    
    /// Texture for displaying the simulation
    pub display_texture: Texture,
    pub display_texture_view: TextureView,
    
    /// Shader handles
    pub setter_shader: Handle<Shader>,
    pub move_shader: Handle<Shader>,
    pub deposit_shader: Handle<Shader>,
    pub diffusion_shader: Handle<Shader>,
    
    /// Pipeline IDs
    pub setter_pipeline_id: CachedComputePipelineId,
    pub move_pipeline_id: CachedComputePipelineId,
    pub deposit_pipeline_id: CachedComputePipelineId,
    pub diffusion_pipeline_id: CachedComputePipelineId,
    
    /// Bind group layout and bind groups
    pub compute_bind_group_layout: BindGroupLayout,
    pub compute_bind_group_a: BindGroup,
    pub compute_bind_group_b: BindGroup,
    
    /// Current frame number (used for ping-pong rendering)
    pub frame_num: u32,
}

/// Resource to track the status of pipeline creation
#[derive(Resource, Default)]
pub struct PipelineStatus {
    pub setter_ready: bool,
    pub move_ready: bool,
    pub deposit_ready: bool,
    pub diffusion_ready: bool,
    pub all_ready: bool,
}

/// Simulation settings
pub mod simulation_settings {
    pub const WIDTH: u32 = 1024;
    pub const HEIGHT: u32 = 1024;
    pub const NUM_PARTICLES: u32 = 1_000_000;
    pub const WORK_GROUP_SIZE: u32 = 8;
    pub const DECAY_FACTOR: f32 = 0.99;
}