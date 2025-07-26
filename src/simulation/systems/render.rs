use bevy::prelude::{default, Commands, Res, ResMut, Sprite, Transform};
use bevy::asset::{AssetServer, Assets, RenderAssetUsages};
use bevy::image::Image;
use bevy::render::render_resource::{AddressMode, BindGroupEntry, BindingResource, BindingType, BufferBinding, BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages, Extent3d, FilterMode, PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, StorageTextureAccess, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDimension};
use bevy::camera::Camera2d;
use bevy::math::{Vec2, Vec3};
use bevy::render::render_asset::RenderAssets;
use bevy::render::texture::GpuImage;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::log::info;
use crate::simulation::constants;
use crate::simulation::render::create_compute_pipeline_id;
use crate::simulation::resources::main::PhysarumInputState;
use crate::simulation::resources::render::{PhysarumBindGroups, PhysarumBuffers, PhysarumImages, PhysarumPipeline, PhysarumSampler, PhysarumSimulationSettings};
use crate::simulation::utils::{binding_entry, create_particles_buffer, load_parameters};

pub fn render_setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let (width, height) = (constants::WIDTH, constants::HEIGHT);

    let mut display_image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    display_image.texture_descriptor.usage |=
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let display_texture = images.add(display_image);

    // Create other textures for the simulation (A and B)
    let mut trail_texture = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &vec![0; (width * height * 2) as usize],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    trail_texture.texture_descriptor.usage |=
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let texture_a = images.add(trail_texture.clone());
    let texture_b = images.add(trail_texture);

    // Insert the image handles as a resource
    commands.insert_resource(PhysarumImages {
        texture_a,
        texture_b,
        display_texture: display_texture.clone(), // Clone the handle for the resource
    });

    commands.spawn(Camera2d);
    commands.spawn((
        Sprite {
            image: display_texture.clone(),
            custom_size: Some(Vec2::new(
                constants::WIDTH as f32,
                constants::HEIGHT as f32,
            )),
            ..default()
        },
        Transform::from_scale(Vec3::splat(constants::DISPLAY_FACTOR as f32)),
    ));
}

fn create_bind_group_entries<'a>(
    read_texture_view: &'a TextureView,
    write_texture_view: &'a TextureView,
    display_texture_view: &'a TextureView,
    buffers: &'a PhysarumBuffers,
    sampler: &'a Sampler,
) -> [BindGroupEntry<'a>; 8] {
    [
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(read_texture_view),
        },
        BindGroupEntry {
            binding: 6,
            resource: BindingResource::Sampler(sampler),
        },
        BindGroupEntry {
            binding: 1,
            resource: BindingResource::TextureView(write_texture_view),
        },
        BindGroupEntry {
            binding: 2,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buffers.particles_buffer,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry {
            binding: 3,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buffers.counter_buffer,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry {
            binding: 4,
            resource: BindingResource::TextureView(&display_texture_view),
        },
        BindGroupEntry {
            binding: 5,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buffers.params_buffer,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry {
            binding: 10,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buffers.uniform_buffer,
                offset: 0,
                size: None,
            }),
        },
    ]
}

pub fn prepare_bind_groups(
    mut commands: Commands,
    pipeline: Res<PhysarumPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Res<PhysarumImages>,
    sampler: Res<PhysarumSampler>,
    buffers: Res<PhysarumBuffers>,
    render_device: Res<RenderDevice>,
) {
    let view_a = gpu_images
        .get(&images.texture_a)
        .expect("Texture A not found");

    let view_b = gpu_images
        .get(&images.texture_b)
        .expect("Texture B not found");

    let display_view = gpu_images
        .get(&images.display_texture)
        .expect("Display texture not found");

    let compute_bind_group_a = render_device.create_bind_group(
        Some("Compute Bind Group A"),
        &pipeline.compute_bind_group_layout,
        // &entries_fn(&view_a.texture_view, &view_b.texture_view)
        &create_bind_group_entries(
            &view_a.texture_view,
            &view_b.texture_view,
            &display_view.texture_view,
            &buffers,
            &sampler.0,
        ),
    );

    let compute_bind_group_b = render_device.create_bind_group(
        Some("Compute Bind Group B"),
        &pipeline.compute_bind_group_layout,
        &create_bind_group_entries(
            &view_b.texture_view,
            &view_a.texture_view,
            &display_view.texture_view,
            &buffers,
            &sampler.0,
        ),
    );

    commands.insert_resource(PhysarumBindGroups([
        compute_bind_group_a,
        compute_bind_group_b,
    ]))
}

pub fn init_physarum_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut pipeline_cache: ResMut<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    // Create a sampler for texture reads
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        label: "Sampler".into(),
        ..Default::default()
    });

    commands.insert_resource(PhysarumSampler(sampler));

    // Create the initial buffers
    let particles_buffer = create_particles_buffer(&render_device);

    // Simulation parameters (start at 0)
    let index = 0;
    let point_settings = load_parameters(index);

    commands.insert_resource(PhysarumSimulationSettings {
        index,
        point_settings
    });

    let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Simulation Params Buffer"),
        contents: bytemuck::bytes_of(&point_settings),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });



    // Create uniform buffer for simulation parameters
    let uniform_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Uniform Buffer"),
        size: size_of::<[u32; 3]>() as u64, // width, height, decay/deposit factor
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Write initial data to uniform buffer
    // Make sure all values are consistently u32 to match shader expectations
    let uniform_data = [
        constants::WIDTH,
        constants::HEIGHT,
        constants::DECAY_FACTOR.to_bits(),
    ];
    render_queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));

    let counter_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Counter Buffer"),
        size: (constants::WIDTH * constants::HEIGHT * 4) as u64, // 4 bytes per u32
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(PhysarumBuffers {
        counter_buffer,
        particles_buffer,
        uniform_buffer,
        params_buffer,
    });

    let setter_shader = asset_server.load("shaders/setter.wgsl");
    let move_shader = asset_server.load("shaders/move.wgsl");
    let deposit_shader = asset_server.load("shaders/deposit.wgsl");
    let diffusion_shader = asset_server.load("shaders/diffusion.wgsl");

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
            binding_entry(6, BindingType::Sampler(SamplerBindingType::NonFiltering)), // Trail Sampler
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

    // Process the pipeline queue - ensure this happens after all pipelines are queued
    pipeline_cache.process_queue();
    info!("Pipeline queue processed - pipelines are now being compiled");

    commands.insert_resource(PhysarumPipeline {
        compute_bind_group_layout,
        setter_pipeline_id,
        move_pipeline_id,
        deposit_pipeline_id,
        diffusion_pipeline_id,
    });
}

/// Update simulation parameters in the render world
pub fn update_simulation_params(
    mut input_state: ResMut<PhysarumInputState>,
    buffers: Res<PhysarumBuffers>,
    render_queue: Res<RenderQueue>,
    mut simulation_settings: ResMut<PhysarumSimulationSettings>,
) {
    if input_state.settings_changed {
        simulation_settings.index = input_state.new_index;
        simulation_settings.point_settings = load_parameters(simulation_settings.index);

        let params_bytes = bytemuck::bytes_of(&simulation_settings.point_settings);
        render_queue.write_buffer(&buffers.params_buffer, 0, params_bytes);

        // Reset the flag
        input_state.settings_changed = false;
    }
}
