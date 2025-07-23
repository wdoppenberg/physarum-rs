use bevy::{prelude::*, render::{render_resource::*, renderer::{RenderDevice, RenderQueue}}};
use std::borrow::Cow;
use bevy::render::render_asset::RenderAssetUsages;
use rand::{random, Rng};

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

/// Resources for our simulation
#[derive(Resource)]
struct PhysarumSimulation {
    point_cursor_index: usize,

    // Buffers for GPU data
    particles_buffer: Buffer,
    simulation_params_buffer: Buffer,
    uniform_buffer: Buffer,
    counter_buffer: Buffer,

    // We use two textures (ping-pong) for the trail map to read from one
    // while writing to the other.
    trail_texture_a: Texture,
    trail_texture_view_a: TextureView,
    trail_texture_b: Texture,
    trail_texture_view_b: TextureView,

    // This texture holds the final image to be displayed.
    display_texture: Texture,
    display_texture_view: TextureView,

    // Compute pipelines for each simulation step
    setter_pipeline: ComputePipeline,
    move_pipeline: ComputePipeline,
    deposit_pipeline: ComputePipeline,
    diffusion_pipeline: ComputePipeline,

    // Bind groups to link our buffers and textures to the shaders
    compute_bind_group_a: BindGroup,
    compute_bind_group_b: BindGroup,

    // Keep track of the current frame for ping-ponging
    frame_num: u64,
}

/// Component to display our simulation texture
#[derive(Component)]
struct PhysarumDisplay {
    image_handle: Handle<Image>,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Physarum Simulation".into(),
                resolution: (simulation_settings::WIDTH as f32, simulation_settings::HEIGHT as f32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, handle_input)
        .add_systems(Update, update_simulation)
        .run();
}

/// Setup our simulation resources
fn setup(mut commands: Commands, render_device: Res<RenderDevice>, render_queue: Res<RenderQueue>, mut images: ResMut<Assets<Image>>) {
    // Create a sampler for texture reads
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..Default::default()
    });

    // Create the initial buffers
    let particles_buffer = create_particles_buffer(&render_device);
    let simulation_params_buffer = create_simulation_params_buffer(&render_device, 0);

    // Create uniform buffer for simulation parameters
    let uniform_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Uniform Buffer"),
        size: std::mem::size_of::<[u32; 3]>() as u64, // width, height, decay/deposit factor
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let counter_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Counter Buffer"),
        size: (simulation_settings::WIDTH * simulation_settings::HEIGHT * 4) as u64, // 4 bytes per u32
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create textures for simulation and display
    let (trail_texture_a, trail_texture_view_a) = create_trail_texture(&render_device, "Trail Texture A");
    let (trail_texture_b, trail_texture_view_b) = create_trail_texture(&render_device, "Trail Texture B");
    let (display_texture, display_texture_view) = create_display_texture(&render_device);

    let setter_shader;
    let move_shader;
    let deposit_shader;
    let diffusion_shader;

    // Load shader code
    unsafe {
        setter_shader = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Setter Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/setter.wgsl"))),
        });

        move_shader = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Move Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/move.wgsl"))),
        });

        deposit_shader = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Deposit Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/deposit.wgsl"))),
        });

        diffusion_shader = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Diffusion Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/diffusion.wgsl"))),
        });
    }

    // Define the binding layout
    let compute_bind_group_layout =
        render_device.create_bind_group_layout(
            Some("Compute Bind Group Layout"),
            &[
                binding_entry(
                    0,
                    BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ), // Trail Read
                binding_entry(
                    0,
                    BindingType::Sampler(SamplerBindingType::Filtering),
                ), // Trail Sampler
                binding_entry(
                    1,
                    BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::R32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                ), // Trail Write
                binding_entry(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ), // Particles
                binding_entry(
                    3,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ), // Counter
                binding_entry(
                    4,
                    BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba8Unorm,
                        access: StorageTextureAccess::WriteOnly,
                    },
                ), // Display FBO
                binding_entry(
                    5,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ), // Point Params
                binding_entry(
                    10,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ), // Uniforms
            ],
        );

    // Write initial data to uniform buffer
    let uniform_data = [
        simulation_settings::WIDTH,
        simulation_settings::HEIGHT,
        simulation_settings::DECAY_FACTOR.to_bits(),
    ];
    render_queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));

    // Create bind groups
    let compute_bind_group_a = create_bind_group(
        &render_device,
        &compute_bind_group_layout,
        &trail_texture_view_a,
        &trail_texture_view_b,
        &particles_buffer,
        &counter_buffer,
        &display_texture_view,
        &simulation_params_buffer,
        &uniform_buffer,
        &sampler,
    );

    let compute_bind_group_b = create_bind_group(
        &render_device,
        &compute_bind_group_layout,
        &trail_texture_view_b,
        &trail_texture_view_a,
        &particles_buffer,
        &counter_buffer,
        &display_texture_view,
        &simulation_params_buffer,
        &uniform_buffer,
        &sampler,
    );

    // Create pipeline layout
    let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipelines
    let setter_pipeline = create_compute_pipeline(&render_device, &pipeline_layout, &setter_shader, "main");
    let move_pipeline = create_compute_pipeline(&render_device, &pipeline_layout, &move_shader, "main");
    let deposit_pipeline = create_compute_pipeline(&render_device, &pipeline_layout, &deposit_shader, "main");
    let diffusion_pipeline = create_compute_pipeline(&render_device, &pipeline_layout, &diffusion_shader, "main");

    // Store all resources
    commands.insert_resource(PhysarumSimulation {
        point_cursor_index: 0,
        particles_buffer,
        simulation_params_buffer,
        uniform_buffer,
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
    });

    // Create a sprite to display our simulation
    let size = Extent3d {
        width: simulation_settings::WIDTH,
        height: simulation_settings::HEIGHT,
        depth_or_array_layers: 1,
    };

    // Create a 2D texture that will be updated from the compute shader
    let mut display_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::all()
    );
    display_image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let display_image_handle = images.add(display_image);

    // Spawn a sprite that uses our texture
    commands.spawn((
        Sprite::from_image(display_image_handle.clone()),
        PhysarumDisplay {
            image_handle: display_image_handle
        }
    ));

    // Setup 2D camera
    commands.spawn(Camera2d::default());
}

/// Handle keyboard input to change simulation parameters
fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut simulation: ResMut<PhysarumSimulation>,
    render_queue: Res<RenderQueue>
) {
    let mut changed = false;

    if keys.just_pressed(KeyCode::ArrowRight) || keys.just_pressed(KeyCode::ArrowUp) {
        simulation.point_cursor_index = (simulation.point_cursor_index + 1) % NUMBER_OF_BASE_POINTS;
        changed = true;
    }

    if keys.just_pressed(KeyCode::ArrowLeft) || keys.just_pressed(KeyCode::ArrowDown) {
        simulation.point_cursor_index = (simulation.point_cursor_index + NUMBER_OF_BASE_POINTS - 1)
            % NUMBER_OF_BASE_POINTS;
        changed = true;
    }

    if changed {
        // If the preset changed, update the parameters on the GPU
        let params = load_parameters(simulation.point_cursor_index);
        let params_bytes = bytemuck::bytes_of(&params);
        render_queue.write_buffer(
            &simulation.simulation_params_buffer,
            0,
            params_bytes,
        );
    }
}

/// Update the simulation each frame
fn update_simulation(
    mut simulation: ResMut<PhysarumSimulation>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut images: ResMut<Assets<Image>>,
    query: Query<&PhysarumDisplay>,
) {
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Physarum Compute Encoder"),
    });

    // Select the correct bind group for this frame to ping-pong the textures
    let (read_bind_group, write_bind_group) = if simulation.frame_num % 2 == 0 {
        (&simulation.compute_bind_group_a, &simulation.compute_bind_group_b)
    } else {
        (&simulation.compute_bind_group_b, &simulation.compute_bind_group_a)
    };

    // Dispatch Setter Shader: Clears the counter buffer
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Setter Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&simulation.setter_pipeline);
        compute_pass.set_bind_group(0, read_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Dispatch Move Shader: Updates particle positions
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Move Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&simulation.move_pipeline);
        compute_pass.set_bind_group(0, read_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::NUMBER_OF_PARTICLES
                / (128 * simulation_settings::PARTICLE_PARAMETERS_COUNT),
            1,
            1,
        );
    }

    // Dispatch Deposit Shader: Deposits trails from particles
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Deposit Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&simulation.deposit_pipeline);
        // This pass writes to the *other* texture, so we use the `write_bind_group`
        compute_pass.set_bind_group(0, write_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Dispatch Diffusion Shader: Blurs and fades the trail map
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Diffusion Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&simulation.diffusion_pipeline);
        // This pass reads the texture written by the deposit shader and writes back to the original
        compute_pass.set_bind_group(0, write_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Copy the display texture to our sprite's image
    if let Ok(physarum_display) = query.get_single() {
        if let Some(image) = images.get_mut(&physarum_display.image_handle) {
            // In Bevy 0.16, we need to use the GPU texture directly
            let gpu_image = render_device.create_texture(&TextureDescriptor {
                label: Some("Display Image Copy"),
                size: Extent3d {
                    width: simulation_settings::WIDTH,
                    height: simulation_settings::HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            
            // In Bevy 0.16, we need to use a simpler approach
            // Instead of trying to copy textures directly, we'll just update the image data
            
            // Update the image size to match the simulation
            image.resize(Extent3d {
                width: simulation_settings::WIDTH,
                height: simulation_settings::HEIGHT,
                depth_or_array_layers: 1,
            });
            
            // Create a new texture with the same size
            let new_texture = render_device.create_texture(&TextureDescriptor {
                label: Some("Updated Display Texture"),
                size: Extent3d {
                    width: simulation_settings::WIDTH,
                    height: simulation_settings::HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            
            // Copy from simulation texture to the new texture
            encoder.copy_texture_to_texture(
                simulation.display_texture.as_image_copy(),
                new_texture.as_image_copy(),
                Extent3d {
                    width: simulation_settings::WIDTH,
                    height: simulation_settings::HEIGHT,
                    depth_or_array_layers: 1,
                },
            );
            
            // Update the image data
            image.texture_descriptor.size = Extent3d {
                width: simulation_settings::WIDTH,
                height: simulation_settings::HEIGHT,
                depth_or_array_layers: 1,
            };
            image.texture_descriptor.format = TextureFormat::Rgba8Unorm;
        }
    }

    // Submit the work to the GPU
    render_queue.submit(Some(encoder.finish()));
    simulation.frame_num += 1;
}

// ----------------- HELPER FUNCTIONS -----------------

/// Creates the initial buffer for particles with random positions
fn create_particles_buffer(render_device: &RenderDevice) -> Buffer {
    let mut particles = vec![
        0u16;
        (simulation_settings::NUMBER_OF_PARTICLES * simulation_settings::PARTICLE_PARAMETERS_COUNT)
            as usize
    ];
    let mut rng = rand::thread_rng();

    for i in 0..simulation_settings::NUMBER_OF_PARTICLES as usize {
        let x = rng.gen_range(0.0..simulation_settings::WIDTH as f32)
            / simulation_settings::WIDTH as f32;
        let y = rng.gen_range(0.0..simulation_settings::HEIGHT as f32)
            / simulation_settings::HEIGHT as f32;
        let angle = rng.gen::<f32>();
        let species = rng.gen::<f32>(); // Or other parameter

        let float_as_u16 = |f: f32| (f.clamp(0.0, 1.0) * 65535.0).round() as u16;

        let base_idx = i * simulation_settings::PARTICLE_PARAMETERS_COUNT as usize;
        particles[base_idx + 0] = float_as_u16(x);
        particles[base_idx + 1] = float_as_u16(y);
        particles[base_idx + 2] = float_as_u16(angle);
        particles[base_idx + 3] = float_as_u16(species);
    }

    let particles_bytes = bytemuck::cast_slice(&particles);
    render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Particles Buffer"),
        contents: particles_bytes,
        usage: BufferUsages::STORAGE,
    })
}

/// Creates the buffer for simulation parameters and initializes it
fn create_simulation_params_buffer(render_device: &RenderDevice, index: usize) -> Buffer {
    let params = load_parameters(index);
    let params_bytes = bytemuck::bytes_of(&params);
    render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Simulation Parameters Buffer"),
        contents: params_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// Creates a single trail map texture and its view
fn create_trail_texture(render_device: &RenderDevice, label: &str) -> (Texture, TextureView) {
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: simulation_settings::WIDTH,
            height: simulation_settings::HEIGHT,
            depth_or_array_layers: 1,
        },
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

/// Creates the final display texture and its view
fn create_display_texture(render_device: &RenderDevice) -> (Texture, TextureView) {
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("Display Texture"),
        size: Extent3d {
            width: simulation_settings::WIDTH,
            height: simulation_settings::HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

/// Loads a parameter set from the matrix
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
    render_device: &RenderDevice,
    layout: &PipelineLayout,
    module: &ShaderModule,
    entry_point: &str,
) -> ComputePipeline {
    // NOTE: This is a placeholder implementation to allow the code to compile
    // In Bevy 0.16, the pipeline creation API has changed significantly
    // A proper implementation would need to:
    // 1. Create a shader using Shader::from_wgsl or similar
    // 2. Create a ComputePipelineDescriptor with the appropriate fields
    // 3. Use a PipelineCache to create the pipeline
    
    // For now, we'll just create a dummy pipeline
    // This will compile but won't work correctly at runtime
    // You'll need to replace this with a proper implementation
    
    // Create a dummy pipeline
    // In a real application, you would use code like:
    /*
    let shader = Shader::from_wgsl(
        include_str!("../shaders/setter.wgsl"),
        "setter.wgsl",
    );
    
    let descriptor = ComputePipelineDescriptor {
        label: Some(Cow::from(entry_point)),
        layout: vec![],  // Use auto layout
        push_constant_ranges: vec![],
        shader: shader_handle,  // You need to get a Handle<Shader>
        shader_defs: vec![],
        entry_point: Cow::Borrowed(entry_point),
        zero_initialize_workgroup_memory: false,
    };
    
    // Use a PipelineCache to create the pipeline
    pipeline_cache.get_compute_pipeline(descriptor).unwrap()
    */
    
    // Return a placeholder
    unimplemented!("Pipeline creation needs to be reimplemented for Bevy 0.16")
}

fn binding_entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn create_bind_group(
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
                binding: 0,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(write_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: particles_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: counter_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(display_view),
            },
            BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 10,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    )
}
