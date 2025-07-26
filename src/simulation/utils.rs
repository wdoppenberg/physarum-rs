use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderDevice;
use rand::Rng;

use crate::points_basematrix::{NUMBER_OF_BASE_POINTS, PARAMETERS_MATRIX};
use crate::simulation::constants;
use crate::simulation::resources::render::PointSettings;
/// Create a buffer containing the initial particle positions
pub fn create_particles_buffer(render_device: &RenderDevice) -> Buffer {
    let mut rng = rand::rng();
    let mut initial_particle_data = Vec::with_capacity(2 * constants::NUM_PARTICLES as usize);

    for _ in 0..constants::NUM_PARTICLES {
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

/// Pack two f32 values into a u32 using 2x16 unorm format
fn pack_2x16_unorm(x: f32, y: f32) -> u32 {
    let x_u16 = (x.clamp(0.0, 1.0) * 65535.0) as u16;
    let y_u16 = (y.clamp(0.0, 1.0) * 65535.0) as u16;
    (x_u16 as u32) | ((y_u16 as u32) << 16)
}