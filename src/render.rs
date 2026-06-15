use crate::buffers::UniformData;
use crate::resources::config::PhysarumConfig;
use crate::resources::input::PhysarumInputState;
use crate::resources::render::{PhysarumBindGroups, PhysarumBuffers, PhysarumPipeline};
use bevy::log::{debug, info};
use bevy::prelude::*;
use bevy::render::render_graph::{self, RenderLabel};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderQueue};

/// Create a compute pipeline ID and queue it for creation
pub fn create_compute_pipeline_id(
    pipeline_cache: &mut PipelineCache,
    layout: &BindGroupLayoutDescriptor,
    shader: &Handle<Shader>,
    entry_point: &str,
) -> CachedComputePipelineId {
    let pipeline_descriptor = ComputePipelineDescriptor {
        label: None,
        layout: vec![layout.clone()],
        push_constant_ranges: Vec::new(),
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some(entry_point.to_owned().into()),
        zero_initialize_workgroup_memory: false,
    };

    pipeline_cache.queue_compute_pipeline(pipeline_descriptor)
}

/// Check if a compute pipeline is ready
pub fn check_pipeline_ready(
    pipeline_cache: &PipelineCache,
    pipeline_id: CachedComputePipelineId,
) -> bool {
    matches!(
        pipeline_cache.get_compute_pipeline_state(pipeline_id),
        CachedPipelineState::Ok(_)
    )
}

/// Label for the Physarum simulation node in the render graph
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct PhysarumSimulationLabel;

#[derive(Debug)]
enum UpdateState {
    Ping,
    Pong,
}

/// State of the Physarum simulation
enum PhysarumSimulationState {
    /// Waiting for pipelines to load
    Loading,
    /// Initializing the simulation
    Init,
    /// Running the simulation with ping-pong between textures
    Update(UpdateState),
}

/// Render graph node for the Physarum simulation
pub struct PhysarumSimulationNode {
    state: PhysarumSimulationState,
}

#[derive(Clone, Copy)]
struct ReactiveTuning {
    deposit_value: f32,
    move_value: f32,
    diffusion_value: f32,
    action_sigma: f32,
    liveliness: f32,
}

fn compute_reactive_tuning(config: &PhysarumConfig, input: &PhysarumInputState) -> ReactiveTuning {
    let mut bass = 0.0;
    let mut mid = 0.0;
    let mut treble = 0.0;
    let mut beat = 0.0;

    if config.audio_reactive_enabled {
        let gain = config.audio_reactive_gain.max(0.0);
        bass = (input.audio_bass * gain * config.audio_bass_influence).clamp(0.0, 2.0);
        mid = (input.audio_mid * gain * config.audio_mid_influence).clamp(0.0, 2.0);
        treble = (input.audio_treble * gain * config.audio_treble_influence).clamp(0.0, 2.0);
        beat = (input.audio_beat * gain * config.audio_beat_influence).clamp(0.0, 2.5);
    }

    let energy = (0.50 * bass + 0.30 * mid + 0.20 * treble + 0.40 * beat).clamp(0.0, 2.0);
    let liveliness = (1.0 + config.liveliness_boost.max(0.0) * energy).clamp(1.0, 3.5);

    let deposit_value =
        (config.deposit_factor * (1.0 + 0.45 * bass + 0.35 * beat) * liveliness).clamp(0.0, 16.0);
    let move_value = (config.pixel_scale_factor
        * (1.0 + 0.35 * mid + 0.25 * treble + 0.20 * beat)
        * (1.0 + 0.12 * (liveliness - 1.0)))
        .clamp(0.0, 8.0);
    let diffusion_value =
        (config.decay_factor - 0.03 * treble - 0.02 * mid - 0.025 * beat).clamp(0.82, 0.9995);
    let action_sigma =
        (config.action_area_size_sigma * (1.0 + 0.50 * bass + 0.40 * beat)).clamp(0.03, 2.5);

    ReactiveTuning {
        deposit_value,
        move_value,
        diffusion_value,
        action_sigma,
        liveliness,
    }
}

fn build_uniform_data(
    config: &PhysarumConfig,
    input: &PhysarumInputState,
    tuning: ReactiveTuning,
    pass_value: f32,
) -> UniformData {
    let beat_spawn = if config.audio_reactive_enabled && input.audio_beat > 0.55 {
        1
    } else {
        0
    };

    UniformData {
        width: config.width,
        height: config.height,
        value: pass_value,
        color_mode: config.color_mode,
        num_particles: config.num_particles,
        time: input.time,
        action_area_size_sigma: tuning.action_sigma,
        action_x: input.action_x,
        action_y: input.action_y,
        move_bias_action_x: input.move_bias_action_x,
        move_bias_action_y: input.move_bias_action_y,
        l2_action: input.l2_action,
        spawn_particles: input.spawn_particles.max(beat_spawn),
        spawn_fraction: input.spawn_fraction,
        random_spawn_number: input.random_spawn_number,
        num_boids: input.num_boids,
        audio_level: input.audio_level,
        audio_bass: input.audio_bass,
        audio_mid: input.audio_mid,
        audio_treble: input.audio_treble,
        audio_beat: input.audio_beat,
        dark_profile_enabled: u32::from(config.dark_profile_enabled),
        dark_max_luminance: config.dark_max_luminance.clamp(0.05, 1.0),
        dark_contrast: config.dark_contrast.clamp(0.5, 3.0),
        dark_black_lift: config.dark_black_lift.clamp(0.0, 0.2),
        liveliness: tuning.liveliness,
    }
}

fn write_uniform(queue: &RenderQueue, uniform_buffer: &Buffer, uniform_data: &UniformData) {
    let mut buffer = encase::UniformBuffer::new(Vec::new());
    buffer.write(uniform_data).unwrap();
    queue.write_buffer(uniform_buffer, 0, &buffer.into_inner());
}

impl Default for PhysarumSimulationNode {
    fn default() -> Self {
        Self {
            state: PhysarumSimulationState::Loading,
        }
    }
}

impl render_graph::Node for PhysarumSimulationNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<PhysarumPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // Check the current state and transition if needed
        match self.state {
            PhysarumSimulationState::Loading => {
                // Check if all pipelines are ready
                let setter_ready =
                    check_pipeline_ready(pipeline_cache, pipeline.setter_pipeline_id);
                let move_ready = check_pipeline_ready(pipeline_cache, pipeline.move_pipeline_id);
                let deposit_ready =
                    check_pipeline_ready(pipeline_cache, pipeline.deposit_pipeline_id);
                let diffusion_ready =
                    check_pipeline_ready(pipeline_cache, pipeline.diffusion_pipeline_id);

                if setter_ready && move_ready && deposit_ready && diffusion_ready {
                    debug!("All pipelines ready, transitioning to Init state");
                    self.state = PhysarumSimulationState::Init;
                }
            }
            PhysarumSimulationState::Init => {
                // After initialization, transition to Update state
                info!("Simulation initialized, starting render.");
                self.state = PhysarumSimulationState::Update(UpdateState::Ping);
            }
            PhysarumSimulationState::Update(UpdateState::Ping) => {
                // Ping-pong between textures
                self.state = PhysarumSimulationState::Update(UpdateState::Pong);
            }
            PhysarumSimulationState::Update(UpdateState::Pong) => {
                // Ping-pong between textures
                self.state = PhysarumSimulationState::Update(UpdateState::Ping);
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<PhysarumPipeline>();
        let [bind_group_a, bind_group_b] = &world.resource::<PhysarumBindGroups>().0;
        let physarum_buffers = world.resource::<PhysarumBuffers>();
        let queue = world.resource::<RenderQueue>();
        let input = world.resource::<PhysarumInputState>();
        let config = world.resource::<PhysarumConfig>();
        let tuning = compute_reactive_tuning(&config, &input);

        match &self.state {
            PhysarumSimulationState::Loading => {}
            PhysarumSimulationState::Init => {
                let mut encoder = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

                if let Some(pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.setter_pipeline_id)
                {
                    let uniform_data =
                        build_uniform_data(&config, &input, tuning, tuning.deposit_value);
                    write_uniform(queue, &physarum_buffers.uniform_buffer, &uniform_data);

                    encoder.set_pipeline(pipeline);
                    encoder.set_bind_group(0, bind_group_a, &[]);
                    encoder.dispatch_workgroups(
                        config.width / config.work_group_size,
                        config.height / config.work_group_size,
                        1,
                    );
                }
            }
            PhysarumSimulationState::Update(swap) => {
                let (deposit_bind_group, diffusion_bind_group) = match swap {
                    UpdateState::Ping => (bind_group_a, bind_group_b),
                    UpdateState::Pong => (bind_group_b, bind_group_a),
                };

                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());

                    // 1. Deposit Pass FIRST (reads counters from previous frame's move pass)
                    if let Some(pipeline) =
                        pipeline_cache.get_compute_pipeline(pipeline.deposit_pipeline_id)
                    {
                        let uniform_data =
                            build_uniform_data(&config, &input, tuning, tuning.deposit_value);
                        write_uniform(queue, &physarum_buffers.uniform_buffer, &uniform_data);
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, deposit_bind_group, &[]);
                        pass.dispatch_workgroups(
                            config.width / config.work_group_size,
                            config.height / config.work_group_size,
                            1,
                        );
                    }

                    // 2. Clear counters AFTER deposit has read them
                    if let Some(pipeline) =
                        pipeline_cache.get_compute_pipeline(pipeline.setter_pipeline_id)
                    {
                        let uniform_data = build_uniform_data(&config, &input, tuning, 0.0);
                        write_uniform(queue, &physarum_buffers.uniform_buffer, &uniform_data);
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, deposit_bind_group, &[]);
                        pass.dispatch_workgroups(
                            config.width / config.work_group_size,
                            config.height / config.work_group_size,
                            1,
                        );
                    }

                    // 3. Move Pass (writes new counts for next frame)
                    if let Some(pipeline) =
                        pipeline_cache.get_compute_pipeline(pipeline.move_pipeline_id)
                    {
                        let uniform_data =
                            build_uniform_data(&config, &input, tuning, tuning.move_value);
                        write_uniform(queue, &physarum_buffers.uniform_buffer, &uniform_data);
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, deposit_bind_group, &[]);

                        // Calculate 2D dispatch to avoid exceeding 65535 limit
                        let total_groups = config.num_particles.div_ceil(128);
                        const MAX_GROUPS_PER_DIM: u32 = 65535;

                        let dispatch_x = std::cmp::min(total_groups, MAX_GROUPS_PER_DIM);
                        let dispatch_y = total_groups.div_ceil(dispatch_x);

                        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                    }
                }

                // 4. Diffusion pass in separate compute pass
                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());

                    if let Some(pipeline) =
                        pipeline_cache.get_compute_pipeline(pipeline.diffusion_pipeline_id)
                    {
                        let uniform_data =
                            build_uniform_data(&config, &input, tuning, tuning.diffusion_value);
                        write_uniform(queue, &physarum_buffers.uniform_buffer, &uniform_data);
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, diffusion_bind_group, &[]);
                        pass.dispatch_workgroups(
                            config.width / config.work_group_size,
                            config.height / config.work_group_size,
                            1,
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
