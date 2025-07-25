use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};

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

#[derive(Resource, Clone)]
pub(crate) struct PhysarumSampler(pub Sampler);

#[derive(Resource, Clone)]
pub(crate) struct PhysarumBuffers {
    pub(crate) uniform_buffer: Buffer,
    pub(crate) counter_buffer: Buffer,
    pub(crate) particles_buffer: Buffer,
    pub(crate) params_buffer: Buffer,
}

#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct PhysarumImages {
    pub(crate) texture_a: Handle<Image>,
    pub(crate) texture_b: Handle<Image>,
    pub(crate) display_texture: Handle<Image>
}

#[derive(Resource)]
pub struct PhysarumPipeline {
    pub(crate) compute_bind_group_layout: BindGroupLayout,
    pub setter_pipeline_id: CachedComputePipelineId,
    pub move_pipeline_id: CachedComputePipelineId,
    pub deposit_pipeline_id: CachedComputePipelineId,
    pub diffusion_pipeline_id: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct PhysarumBindGroups(pub [BindGroup; 2]);


/// Simulation settings
pub mod simulation_settings {
    pub const WIDTH: u32 = 1024;
    pub const HEIGHT: u32 = 1024;
    pub const DISPLAY_FACTOR: u32 = 4;
    pub const NUM_PARTICLES: u32 = 1_000_000;
    pub const WORK_GROUP_SIZE: u32 = 32;
    pub const DECAY_FACTOR: f32 = 0.99;
    pub const PIXEL_SCALE_FACTOR: f32 = 1.0;
    pub const DEPOSIT_FACTOR: f32 = 1.0;
}
