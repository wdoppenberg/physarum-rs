use bevy::prelude::Resource;
use bevy::render::render_resource::{BindGroup, BindGroupLayout, Buffer, CachedComputePipelineId, Sampler};
use bevy::render::extract_resource::ExtractResource;
use bevy::asset::Handle;
use bevy::image::Image;

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

#[derive(Resource)]
pub struct PhysarumSimulationSettings {
    pub(crate) index: usize,
    pub(crate) point_settings: PointSettings
}
