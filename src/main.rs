use nannou::prelude::*;
use nannou::wgpu::{include_wgsl, ToTextureView};

mod points_basematrix;
use points_basematrix::{NUMBER_OF_BASE_POINTS, PARAMETERS_MATRIX};

// Simulation constants, inferred from the C++ code
mod simulation_settings {
    pub const WIDTH: u32 = 1280;
    pub const HEIGHT: u32 = 736;
    // The number of particles was not specified in the original code,
    // so a reasonable value is chosen here.
    pub const NUMBER_OF_PARTICLES: u32 = 1_000_000;
    pub const PARTICLE_PARAMETERS_COUNT: u32 = 4; // x, y, angle, and one other param
    pub const WORK_GROUP_SIZE: u32 = 8;
    pub const DECAY_FACTOR: f32 = 0.97;
    pub const DEPOSIT_FACTOR: f32 = 0.1;
    pub const PIXEL_SCALE_FACTOR: f32 = 1.0;
}

/// A struct that holds the settings for a single "point" or simulation preset.
/// This data is sent to the GPU to control the simulation behavior.
/// The structure is inferred from `ofApp::loadParameters()`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PointSettings {
    default_scaling_factor: f32,
    sensor_distance0: f32,
    sd_exponent: f32,
    sd_amplitude: f32,
    sensor_angle0: f32,
    sa_exponent: f32,
    sa_amplitude: f32,
    rotation_angle0: f32,
    ra_exponent: f32,
    ra_amplitude: f32,
    move_distance0: f32,
    md_exponent: f32,
    md_amplitude: f32,
    sensor_bias1: f32,
    sensor_bias2: f32,
}

/// The main state of our application.
struct Model {
    point_cursor_index: usize,

    // Buffers for GPU data
    particles_buffer: wgpu::Buffer,
    simulation_params_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,

    // We use two textures (ping-pong) for the trail map to read from one
    // while writing to the other.
    trail_texture_a: wgpu::Texture,
    trail_texture_view_a: wgpu::TextureView,
    trail_texture_b: wgpu::Texture,
    trail_texture_view_b: wgpu::TextureView,

    // This texture holds the final image to be displayed.
    display_texture: wgpu::Texture,
    display_texture_view: wgpu::TextureView,

    // Compute pipelines for each simulation step
    setter_pipeline: wgpu::ComputePipeline,
    move_pipeline: wgpu::ComputePipeline,
    deposit_pipeline: wgpu::ComputePipeline,
    diffusion_pipeline: wgpu::ComputePipeline,

    // Bind groups to link our buffers and textures to the shaders
    compute_bind_group_a: wgpu::BindGroup,
    compute_bind_group_b: wgpu::BindGroup,

    frame_num: u64,
}

fn main() {
    nannou::app(model).update(update).run();
}

/// Sets up the initial state of the application.
fn model(app: &App) -> Model {
    // Create the window
    let window_id = app
        .new_window()
        .size(simulation_settings::WIDTH, simulation_settings::HEIGHT)
        .title("Physarum")
        .view(view)
        .event(event)
        .build()
        .unwrap();
    let window = app.window(window_id).unwrap();
    let device = window.device();

    // Load the WGSL shader code.
    let setter_shader = include_wgsl!("../shaders/setter.wgsl");
    let move_shader = include_wgsl!("../shaders/move.wgsl");
    let deposit_shader = include_wgsl!("../shaders/deposit.wgsl");
    let diffusion_shader = include_wgsl!("../shaders/diffusion.wgsl");

    // Create the compute shader modules.
    let setter_module = device.create_shader_module(setter_shader);
    let move_module = device.create_shader_module(move_shader);
    let deposit_module = device.create_shader_module(deposit_shader);
    let diffusion_module = device.create_shader_module(diffusion_shader);

    // Create GPU buffers.
    let particles_buffer = create_particles_buffer(device);
    let simulation_params_buffer = create_simulation_params_buffer(device, 0);
    let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Counter Buffer"),
        size: (simulation_settings::WIDTH * simulation_settings::HEIGHT * 4) as u64, // 4 bytes per u32
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create textures for simulation and display.
    let (trail_texture_a, trail_texture_view_a) = create_trail_texture(device, "Trail Texture A");
    let (trail_texture_b, trail_texture_view_b) = create_trail_texture(device, "Trail Texture B");
    let (display_texture, display_texture_view) = create_display_texture(device);

    // This layout defines the resources (buffers, textures) our shaders can access.
    let compute_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // All the bindings used across our different compute shaders.
                // Not all shaders will use all bindings.
                binding_entry(
                    0,
                    wgpu::BindingType::StorageTexture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        format: wgpu::TextureFormat::R32Float,
                        access: wgpu::StorageTextureAccess::ReadOnly,
                    },
                ), // Trail Read
                binding_entry(
                    1,
                    wgpu::BindingType::StorageTexture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        format: wgpu::TextureFormat::R32Float,
                        access: wgpu::StorageTextureAccess::WriteOnly,
                    },
                ), // Trail Write
                binding_entry(
                    2,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ), // Particles
                binding_entry(
                    3,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ), // Counter
                binding_entry(
                    4,
                    wgpu::BindingType::StorageTexture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        access: wgpu::StorageTextureAccess::WriteOnly,
                    },
                ), // Display FBO
                binding_entry(
                    5,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ), // Sim Params
            ],
        });

    // Create two bind groups for our ping-pong texture setup.
    let compute_bind_group_a = create_bind_group(
        device,
        &compute_bind_group_layout,
        &trail_texture_view_a,
        &trail_texture_view_b,
        &particles_buffer,
        &counter_buffer,
        &display_texture_view,
        &simulation_params_buffer,
    );
    let compute_bind_group_b = create_bind_group(
        device,
        &compute_bind_group_layout,
        &trail_texture_view_b,
        &trail_texture_view_a,
        &particles_buffer,
        &counter_buffer,
        &display_texture_view,
        &simulation_params_buffer,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create the compute pipelines.
    let setter_pipeline = create_compute_pipeline(device, &pipeline_layout, &setter_module, "main");
    let move_pipeline = create_compute_pipeline(device, &pipeline_layout, &move_module, "main");
    let deposit_pipeline =
        create_compute_pipeline(device, &pipeline_layout, &deposit_module, "main");
    let diffusion_pipeline =
        create_compute_pipeline(device, &pipeline_layout, &diffusion_module, "main");

    Model {
        point_cursor_index: 0,
        particles_buffer,
        simulation_params_buffer,
        counter_buffer,
        trail_texture_a,
        trail_texture_view_a,
        trail_texture_b,
        trail_texture_view_b,
        display_texture,
        display_texture_view,
        setter_pipeline,
        move_pipeline,
        deposit_pipeline,
        diffusion_pipeline,
        compute_bind_group_a,
        compute_bind_group_b,
        frame_num: 0,
    }
}

/// The main simulation loop.
fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.device();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    // Select the correct bind group for this frame to ping-pong the textures.
    let (read_bind_group, write_bind_group) = if model.frame_num % 2 == 0 {
        (&model.compute_bind_group_a, &model.compute_bind_group_b)
    } else {
        (&model.compute_bind_group_b, &model.compute_bind_group_a)
    };

    // Dispatch Setter Shader: Clears the counter buffer.
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Setter Pass"),
        });
        compute_pass.set_pipeline(&model.setter_pipeline);
        compute_pass.set_bind_group(0, read_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Dispatch Move Shader: Updates particle positions.
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Move Pass"),
        });
        compute_pass.set_pipeline(&model.move_pipeline);
        compute_pass.set_bind_group(0, read_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::NUMBER_OF_PARTICLES
                / (128 * simulation_settings::PARTICLE_PARAMETERS_COUNT),
            1,
            1,
        );
    }

    // Dispatch Deposit Shader: Deposits trails from particles.
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Deposit Pass"),
        });
        compute_pass.set_pipeline(&model.deposit_pipeline);
        // This pass writes to the *other* texture, so we use the `write_bind_group`.
        compute_pass.set_bind_group(0, write_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Dispatch Diffusion Shader: Blurs and fades the trail map.
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Diffusion Pass"),
        });
        compute_pass.set_pipeline(&model.diffusion_pipeline);
        // This pass reads the texture written by the deposit shader and writes back to the original.
        compute_pass.set_bind_group(0, write_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Submit the commands to the GPU.
    app.main_window().queue().submit(Some(encoder.finish()));
    model.frame_num += 1;
}

/// Handles window events, like key presses.
fn event(app: &App, model: &mut Model, event: WindowEvent) {
    match event {
        KeyPressed(key) => {
            let mut changed = false;
            match key {
                Key::Right | Key::Up => {
                    model.point_cursor_index =
                        (model.point_cursor_index + 1) % NUMBER_OF_BASE_POINTS;
                    changed = true;
                }
                Key::Left | Key::Down => {
                    model.point_cursor_index = (model.point_cursor_index + NUMBER_OF_BASE_POINTS
                        - 1)
                        % NUMBER_OF_BASE_POINTS;
                    changed = true;
                }
                _ => {}
            }
            if changed {
                // If the preset changed, update the parameters on the GPU.
                let params = load_parameters(model.point_cursor_index);
                let params_bytes = bytemuck::bytes_of(&params);
                app.main_window().queue().write_buffer(
                    &model.simulation_params_buffer,
                    0,
                    params_bytes,
                );
            }
        }
        _ => {}
    }
}

/// Draws the final output to the screen.
fn view(app: &App, model: &Model, frame: Frame) {
    // We don't use nannou's `draw` API directly for the simulation,
    // but we use it here to render our final texture to the window.
    let mut draw = app.draw();
    draw.texture(&model.display_texture_view);

    // Draw to the frame.
    draw.to_frame(app, &frame).unwrap();
}

// ----------------- HELPER FUNCTIONS -----------------

/// Creates the initial buffer for particles with random positions.
fn create_particles_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    let mut particles = vec![
        0u16;
        (simulation_settings::NUMBER_OF_PARTICLES * simulation_settings::PARTICLE_PARAMETERS_COUNT)
            as usize
    ];
    for i in 0..simulation_settings::NUMBER_OF_PARTICLES as usize {
        let x = random_range(0.0, simulation_settings::WIDTH as f32)
            / simulation_settings::WIDTH as f32;
        let y = random_range(0.0, simulation_settings::HEIGHT as f32)
            / simulation_settings::HEIGHT as f32;
        let angle = random::<f32>();
        let species = random::<f32>(); // Or other parameter

        let float_as_u16 = |f: f32| (f.clamp(0.0, 1.0) * 65535.0).round() as u16;

        let base_idx = i * simulation_settings::PARTICLE_PARAMETERS_COUNT as usize;
        particles[base_idx + 0] = float_as_u16(x);
        particles[base_idx + 1] = float_as_u16(y);
        particles[base_idx + 2] = float_as_u16(angle);
        particles[base_idx + 3] = float_as_u16(species);
    }
    let particles_bytes = bytemuck::cast_slice(&particles);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Particles Buffer"),
        contents: particles_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Creates the buffer for simulation parameters and initializes it.
fn create_simulation_params_buffer(device: &wgpu::Device, index: usize) -> wgpu::Buffer {
    let params = load_parameters(index);
    let params_bytes = bytemuck::bytes_of(&params);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Simulation Parameters Buffer"),
        contents: params_bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Creates a single trail map texture and its view.
fn create_trail_texture(device: &wgpu::Device, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = wgpu::TextureBuilder::new()
        .size([simulation_settings::WIDTH, simulation_settings::HEIGHT])
        .format(wgpu::TextureFormat::R32Float)
        .dimension(wgpu::TextureDimension::D2)
        .usage(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING)
        .build(device);
    let view = texture.to_texture_view();
    (texture, view)
}

/// Creates the final display texture and its view.
fn create_display_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = wgpu::TextureBuilder::new()
        .size([simulation_settings::WIDTH, simulation_settings::HEIGHT])
        .format(wgpu::TextureFormat::Rgba8Unorm)
        .usage(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING)
        .dimension(wgpu::TextureDimension::D2)
        .build(device);
    let view = texture.to_texture_view();
    (texture, view)
}

/// Loads a parameter set from the matrix.
fn load_parameters(index: usize) -> PointSettings {
    let row = &points_basematrix::PARAMETERS_MATRIX[index];
    PointSettings {
        default_scaling_factor: row[14],
        sensor_distance0: row[0],
        sd_exponent: row[1],
        sd_amplitude: row[2],
        sensor_angle0: row[3],
        sa_exponent: row[4],
        sa_amplitude: row[5],
        rotation_angle0: row[6],
        ra_exponent: row[7],
        ra_amplitude: row[8],
        move_distance0: row[9],
        md_exponent: row[10],
        md_amplitude: row[11],
        sensor_bias1: row[12],
        sensor_bias2: row[13],
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        module,
        entry_point,
    })
}

fn binding_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    read_view: &wgpu::TextureView,
    write_view: &wgpu::TextureView,
    particles_buffer: &wgpu::Buffer,
    counter_buffer: &wgpu::Buffer,
    display_view: &wgpu::TextureView,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(read_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(write_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: particles_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: counter_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(display_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}
