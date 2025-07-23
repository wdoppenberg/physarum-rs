use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderDevice;
use rand::Rng;

use crate::points_basematrix::{NUMBER_OF_BASE_POINTS, PARAMETERS_MATRIX};
use super::resources::{PointSettings, simulation_settings};

/// Create a buffer containing the initial particle positions
pub fn create_particles_buffer(render_device: &RenderDevice) -> Buffer {
    let mut rng = rand::rng();
    let mut initial_particle_data = Vec::with_capacity(2 * simulation_settings::NUM_PARTICLES as usize);

    for _ in 0..simulation_settings::NUM_PARTICLES {
        // Position (packed as 2x16 unorm)
        let x = rng.random::<f32>();
        let y = rng.random::<f32>();
        let pos_packed = pack_2x16_unorm(x, y);
        initial_particle_data.push(pos_packed);

        // Progress and heading (packed as 2x16 unorm)
        let progress = rng.random::<f32>(); // 0.0 to 1.0
        let heading = rng.random::<f32>() * std::f32::consts::PI * 2.0; // 0 to 2π
        let heading_normalized = heading / (2.0 * std::f32::consts::PI); // Normalize to 0-1
        let progress_heading_packed = pack_2x16_unorm(progress, heading_normalized);
        initial_particle_data.push(progress_heading_packed);
    }

    render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Particle Buffer"),
        contents: bytemuck::cast_slice(&initial_particle_data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Create a buffer containing the simulation parameters
pub fn create_simulation_params_buffer(render_device: &RenderDevice, index: usize) -> Buffer {
    let params = load_parameters(index);
    
    render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Simulation Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Create a texture for the trail
pub fn create_trail_texture(render_device: &RenderDevice, label: &str) -> (Texture, TextureView) {
    let size = Extent3d {
        width: simulation_settings::WIDTH,
        height: simulation_settings::HEIGHT,
        depth_or_array_layers: 1,
    };

    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

/// Create a texture for displaying the simulation
pub fn create_display_texture(render_device: &RenderDevice) -> (Texture, TextureView) {
    let size = Extent3d {
        width: simulation_settings::WIDTH,
        height: simulation_settings::HEIGHT,
        depth_or_array_layers: 1,
    };

    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("Display Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

/// Load parameters from the parameters matrix
pub fn load_parameters(index: usize) -> PointSettings {
    let index = index % NUMBER_OF_BASE_POINTS;
    let params = PARAMETERS_MATRIX[index];
    
    PointSettings {
        default_scaling_factor: 1.0,
        sensor_distance0: params[0],
        sd_exponent: params[1],
        sd_amplitude: params[2],
        sensor_angle0: params[3],
        sa_exponent: params[4],
        sa_amplitude: params[5],
        rotation_angle0: params[6],
        ra_exponent: params[7],
        ra_amplitude: params[8],
        move_distance0: params[9],
        md_exponent: params[10],
        md_amplitude: params[11],
        sensor_bias1: params[12],
        sensor_bias2: params[13],
    }
}

/// Create a binding entry for a bind group layout
pub fn binding_entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// Create a bind group for the compute shader
pub fn create_bind_group(
    render_device: &RenderDevice,
    layout: &BindGroupLayout,
    read_view: &TextureView,
    write_view: &TextureView,
    particles_buffer: &Buffer,
    counter_buffer: &Buffer,
    display_view: &TextureView,
    params_buffer: &Buffer,
    uniform_buffer: &Buffer,
    sampler: &Sampler,
) -> BindGroup {
    render_device.create_bind_group(
        Some("Compute Bind Group"),
        layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(read_view),
            },
            BindGroupEntry {
                binding: 6,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(write_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: particles_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: counter_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(display_view),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: params_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 10,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    )
}

/// Pack two f32 values into a u32 using 2x16 unorm format
fn pack_2x16_unorm(x: f32, y: f32) -> u32 {
    let x_u16 = (x.clamp(0.0, 1.0) * 65535.0) as u16;
    let y_u16 = (y.clamp(0.0, 1.0) * 65535.0) as u16;
    (x_u16 as u32) | ((y_u16 as u32) << 16)
}