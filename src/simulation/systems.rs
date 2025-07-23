use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::mem::size_of;

use super::components::PhysarumDisplay;
use super::resources::{PhysarumSimulation, PipelineStatus, simulation_settings};
use super::utils::*;
use super::render::*;

/// System to initialize resources for the simulation
pub fn setup_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut shaders: ResMut<Assets<Shader>>,
    pipeline_cache: Option<ResMut<PipelineCache>>,
) {
    // If PipelineCache is not available, we can't create pipelines yet
    let mut pipeline_cache = match pipeline_cache {
        Some(cache) => cache,
        None => {
            warn!("PipelineCache not available yet, deferring pipeline creation");
            // Create a minimal simulation resource without pipelines
            commands.insert_resource(PhysarumSimulation {
                point_cursor_index: 0,
                simulation_params_buffer: render_device.create_buffer(&BufferDescriptor {
                    label: Some("Dummy Buffer"),
                    size: 1,
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                uniform_buffer: render_device.create_buffer(&BufferDescriptor {
                    label: Some("Dummy Buffer"),
                    size: 1,
                    usage: BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }),
                counter_buffer: render_device.create_buffer(&BufferDescriptor {
                    label: Some("Dummy Buffer"),
                    size: 1,
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                trail_texture_a: render_device.create_texture(&TextureDescriptor {
                    label: Some("Dummy Texture"),
                    size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }),
                trail_texture_view_a: render_device.create_texture(&TextureDescriptor {
                    label: Some("Dummy Texture"),
                    size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }).create_view(&TextureViewDescriptor::default()),
                trail_texture_b: render_device.create_texture(&TextureDescriptor {
                    label: Some("Dummy Texture"),
                    size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }),
                trail_texture_view_b: render_device.create_texture(&TextureDescriptor {
                    label: Some("Dummy Texture"),
                    size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }).create_view(&TextureViewDescriptor::default()),
                display_texture: render_device.create_texture(&TextureDescriptor {
                    label: Some("Dummy Texture"),
                    size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }),
                display_texture_view: render_device.create_texture(&TextureDescriptor {
                    label: Some("Dummy Texture"),
                    size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }).create_view(&TextureViewDescriptor::default()),
                setter_shader: Handle::default(),
                move_shader: Handle::default(),
                deposit_shader: Handle::default(),
                diffusion_shader: Handle::default(),
                setter_pipeline_id: CachedComputePipelineId::INVALID,
                move_pipeline_id: CachedComputePipelineId::INVALID,
                deposit_pipeline_id: CachedComputePipelineId::INVALID,
                diffusion_pipeline_id: CachedComputePipelineId::INVALID,
                compute_bind_group_layout: render_device.create_bind_group_layout(
                    Some("Dummy Layout"),
                    &[],
                ),
                compute_bind_group_a: render_device.create_bind_group(
                    Some("Dummy Bind Group"),
                    &render_device.create_bind_group_layout(Some("Dummy Layout"), &[]),
                    &[],
                ),
                compute_bind_group_b: render_device.create_bind_group(
                    Some("Dummy Bind Group"),
                    &render_device.create_bind_group_layout(Some("Dummy Layout"), &[]),
                    &[],
                ),
                frame_num: 0,
            });
            
            // Initialize pipeline status
            commands.insert_resource(PipelineStatus::default());
            
            return;
        }
    };
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
        size: size_of::<[u32; 3]>() as u64, // width, height, decay/deposit factor
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
    let (trail_texture_a, trail_texture_view_a) =
        create_trail_texture(&render_device, "Trail Texture A");
    let (trail_texture_b, trail_texture_view_b) =
        create_trail_texture(&render_device, "Trail Texture B");
    let (display_texture, display_texture_view) = create_display_texture(&render_device);

    // Create shader assets
    let setter_shader = shaders.add(Shader::from_wgsl(
        include_str!("../shaders/setter.wgsl"),
        "setter.wgsl",
    ));

    let move_shader = shaders.add(Shader::from_wgsl(
        include_str!("../shaders/move.wgsl"),
        "move.wgsl",
    ));

    let deposit_shader = shaders.add(Shader::from_wgsl(
        include_str!("../shaders/deposit.wgsl"),
        "deposit.wgsl",
    ));

    let diffusion_shader = shaders.add(Shader::from_wgsl(
        include_str!("../shaders/diffusion.wgsl"),
        "diffusion.wgsl",
    ));

    // Define the binding layout
    let compute_bind_group_layout = render_device.create_bind_group_layout(
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
            binding_entry(6, BindingType::Sampler(SamplerBindingType::Filtering)), // Trail Sampler
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
    // Make sure all values are consistently u32 to match shader expectations
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

    // Create pipeline IDs
    // Note: This only queues the pipelines for creation. The actual compilation happens
    // asynchronously, and the pipelines might not be ready immediately
    let setter_pipeline_id = create_compute_pipeline_id(
        &mut pipeline_cache,
        &compute_bind_group_layout,
        &setter_shader,
        "main",
    );

    let move_pipeline_id = create_compute_pipeline_id(
        &mut pipeline_cache,
        &compute_bind_group_layout,
        &move_shader,
        "main",
    );

    let deposit_pipeline_id = create_compute_pipeline_id(
        &mut pipeline_cache,
        &compute_bind_group_layout,
        &deposit_shader,
        "main",
    );

    let diffusion_pipeline_id = create_compute_pipeline_id(
        &mut pipeline_cache,
        &compute_bind_group_layout,
        &diffusion_shader,
        "main",
    );

    // Process the pipeline queue
    pipeline_cache.process_queue();

    // Store all resources
    commands.insert_resource(PhysarumSimulation {
        point_cursor_index: 0,
        simulation_params_buffer,
        uniform_buffer,
        counter_buffer,
        trail_texture_a,
        trail_texture_view_a,
        trail_texture_b,
        trail_texture_view_b,
        display_texture,
        display_texture_view,
        setter_shader,
        move_shader,
        deposit_shader,
        diffusion_shader,
        setter_pipeline_id,
        move_pipeline_id,
        deposit_pipeline_id,
        diffusion_pipeline_id,
        compute_bind_group_layout,
        compute_bind_group_a,
        compute_bind_group_b,
        frame_num: 0,
    });

    // Initialize pipeline status
    commands.insert_resource(PipelineStatus::default());
}

/// System to setup the display for the simulation
pub fn setup_display(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
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
        RenderAssetUsages::all(),
    );
    display_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let display_image_handle = images.add(display_image);

    // Spawn a sprite that uses our texture
    commands.spawn((
        Sprite::from_image(display_image_handle.clone()),
        PhysarumDisplay {
            image_handle: display_image_handle,
        },
    ));

    // Setup 2D camera
    commands.spawn(Camera2d::default());
}

/// Handle keyboard input to change simulation parameters
pub fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut simulation: ResMut<PhysarumSimulation>,
    render_queue: Res<RenderQueue>,
) {
    let mut changed = false;

    if keys.just_pressed(KeyCode::ArrowRight) || keys.just_pressed(KeyCode::ArrowUp) {
        simulation.point_cursor_index = (simulation.point_cursor_index + 1) % crate::points_basematrix::NUMBER_OF_BASE_POINTS;
        changed = true;
    }

    if keys.just_pressed(KeyCode::ArrowLeft) || keys.just_pressed(KeyCode::ArrowDown) {
        simulation.point_cursor_index =
            (simulation.point_cursor_index + crate::points_basematrix::NUMBER_OF_BASE_POINTS - 1) % crate::points_basematrix::NUMBER_OF_BASE_POINTS;
        changed = true;
    }

    if changed {
        // If the preset changed, update the parameters on the GPU
        let params = load_parameters(simulation.point_cursor_index);
        let params_bytes = bytemuck::bytes_of(&params);
        render_queue.write_buffer(&simulation.simulation_params_buffer, 0, params_bytes);
    }
}

/// Update the simulation each frame
pub fn update_simulation(
    pipeline_status: Res<PipelineStatus>,
    mut simulation: ResMut<PhysarumSimulation>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut images: ResMut<Assets<Image>>,
    query: Query<&PhysarumDisplay>,
    pipeline_cache: Option<Res<PipelineCache>>,
) {
    // Check if PipelineCache is available
    let pipeline_cache = match pipeline_cache {
        Some(cache) => cache,
        None => {
            // PipelineCache not available yet
            return;
        }
    };
    
    // Only run the simulation if all pipelines are ready
    if !pipeline_status.all_ready {
        return;
    }
    
    // Check if pipeline IDs are valid
    if simulation.setter_pipeline_id == CachedComputePipelineId::INVALID ||
       simulation.move_pipeline_id == CachedComputePipelineId::INVALID ||
       simulation.deposit_pipeline_id == CachedComputePipelineId::INVALID ||
       simulation.diffusion_pipeline_id == CachedComputePipelineId::INVALID {
        // Pipeline IDs are not valid yet
        return;
    }

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Physarum Compute Encoder"),
    });

    // Select the correct bind group for this frame to ping-pong the textures
    let (read_bind_group, write_bind_group) = if simulation.frame_num % 2 == 0 {
        (
            &simulation.compute_bind_group_a,
            &simulation.compute_bind_group_b,
        )
    } else {
        (
            &simulation.compute_bind_group_b,
            &simulation.compute_bind_group_a,
        )
    };

    // Use the pipeline IDs directly
    let setter_pipeline_id = simulation.setter_pipeline_id;
    let move_pipeline_id = simulation.move_pipeline_id;
    let deposit_pipeline_id = simulation.deposit_pipeline_id;
    let diffusion_pipeline_id = simulation.diffusion_pipeline_id;

    // Dispatch Setter Shader: Clears the counter buffer
    {
        // Update uniform buffer with setter-specific values
        let uniform_data = [
            simulation_settings::WIDTH,
            simulation_settings::HEIGHT,
            0u32, // value: 0 to clear the counter
        ];
        render_queue.write_buffer(
            &simulation.uniform_buffer,
            0,
            bytemuck::cast_slice(&uniform_data),
        );

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Setter Pass"),
            timestamp_writes: None,
        });
        // We already checked that all pipelines are available, so this should never panic
        let pipeline = pipeline_cache
            .get_compute_pipeline(setter_pipeline_id)
            .expect("Setter pipeline should be available");
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, read_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Dispatch Move Shader: Updates particle positions
    {
        // Update uniform buffer with move-specific values
        let uniform_data = [
            simulation_settings::WIDTH,
            simulation_settings::HEIGHT,
            1.0f32.to_bits(), // pixelScaleFactor
        ];
        render_queue.write_buffer(
            &simulation.uniform_buffer,
            0,
            bytemuck::cast_slice(&uniform_data),
        );

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Move Pass"),
            timestamp_writes: None,
        });
        let pipeline = pipeline_cache
            .get_compute_pipeline(move_pipeline_id)
            .expect("Move pipeline should be available");
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, read_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            (simulation_settings::NUM_PARTICLES + 127) / 128,
            1,
            1,
        );
    }

    // Dispatch Deposit Shader: Deposits trail values
    {
        // Update uniform buffer with deposit-specific values
        let uniform_data = [
            simulation_settings::WIDTH,
            simulation_settings::HEIGHT,
            1.0f32.to_bits(), // deposit amount
        ];
        render_queue.write_buffer(
            &simulation.uniform_buffer,
            0,
            bytemuck::cast_slice(&uniform_data),
        );

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Deposit Pass"),
            timestamp_writes: None,
        });
        let pipeline = pipeline_cache
            .get_compute_pipeline(deposit_pipeline_id)
            .expect("Deposit pipeline should be available");
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, write_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Dispatch Diffusion Shader: Applies diffusion to the trail texture
    {
        // Update uniform buffer with diffusion-specific values
        let uniform_data = [
            simulation_settings::WIDTH,
            simulation_settings::HEIGHT,
            simulation_settings::DECAY_FACTOR.to_bits(),
        ];
        render_queue.write_buffer(
            &simulation.uniform_buffer,
            0,
            bytemuck::cast_slice(&uniform_data),
        );

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Diffusion Pass"),
            timestamp_writes: None,
        });
        let pipeline = pipeline_cache
            .get_compute_pipeline(diffusion_pipeline_id)
            .expect("Diffusion pipeline should be available");
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, write_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            simulation_settings::WIDTH / simulation_settings::WORK_GROUP_SIZE,
            simulation_settings::HEIGHT / simulation_settings::WORK_GROUP_SIZE,
            1,
        );
    }

    // Submit the command encoder
    render_queue.submit(std::iter::once(encoder.finish()));

    // Mark the display image as needing an update
    // In Bevy 0.16.1, we don't need to manually copy the texture
    // The render system will handle updating the image based on the texture
    if let Ok(display) = query.single() {
        if let Some(image) = images.get_mut(&display.image_handle) {
            // Mark the image as modified
            image.texture_descriptor.usage |= TextureUsages::COPY_DST;
        }
    }

    // Increment the frame counter
    simulation.frame_num += 1;
}