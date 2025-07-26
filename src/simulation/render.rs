use bevy::prelude::*;
use bevy::render::render_graph::{self, RenderLabel};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderQueue};
use crate::simulation::constants;
use crate::simulation::resources::render::{PhysarumBindGroups, PhysarumBuffers, PhysarumPipeline};

/// Create a compute pipeline ID and queue it for creation
pub fn create_compute_pipeline_id(
    pipeline_cache: &mut PipelineCache,
    layout: &BindGroupLayout,
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
    match pipeline_cache.get_compute_pipeline_state(pipeline_id) {
        CachedPipelineState::Ok(_) => true,
        _ => false,
    }
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

        match &self.state {
            PhysarumSimulationState::Loading => {}
            PhysarumSimulationState::Init => {
                let mut encoder = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

                if let Some(pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.setter_pipeline_id)
                {
                    let uniform_data = [constants::WIDTH, constants::HEIGHT, 0];
                    queue.write_buffer(
                        &physarum_buffers.uniform_buffer,
                        0,
                        bytemuck::cast_slice(&uniform_data),
                    );

                    encoder.set_pipeline(pipeline);
                    encoder.set_bind_group(0, bind_group_a, &[]);
                    encoder.dispatch_workgroups(
                        constants::WIDTH / constants::WORK_GROUP_SIZE,
                        constants::HEIGHT / constants::WORK_GROUP_SIZE,
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
                        let uniform_data = [
                            constants::WIDTH,
                            constants::HEIGHT,
                            constants::DEPOSIT_FACTOR.to_bits(),
                        ];
                        queue.write_buffer(
                            &physarum_buffers.uniform_buffer,
                            0,
                            bytemuck::cast_slice(&uniform_data),
                        );
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, deposit_bind_group, &[]);
                        pass.dispatch_workgroups(
                            constants::WIDTH / constants::WORK_GROUP_SIZE,
                            constants::HEIGHT / constants::WORK_GROUP_SIZE,
                            1,
                        );
                    }

                    // 2. Clear counters AFTER deposit has read them
                    if let Some(pipeline) =
                        pipeline_cache.get_compute_pipeline(pipeline.setter_pipeline_id)
                    {
                        let uniform_data =
                            [constants::WIDTH, constants::HEIGHT, 0];
                        queue.write_buffer(
                            &physarum_buffers.uniform_buffer,
                            0,
                            bytemuck::cast_slice(&uniform_data),
                        );
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, deposit_bind_group, &[]);
                        pass.dispatch_workgroups(
                            constants::WIDTH / constants::WORK_GROUP_SIZE,
                            constants::HEIGHT / constants::WORK_GROUP_SIZE,
                            1,
                        );
                    }

                    // 3. Move Pass (writes new counts for next frame)
                    if let Some(pipeline) =
                        pipeline_cache.get_compute_pipeline(pipeline.move_pipeline_id)
                    {
                        let uniform_data = [
                            constants::WIDTH,
                            constants::HEIGHT,
                            constants::PIXEL_SCALE_FACTOR.to_bits(),
                        ];
                        queue.write_buffer(
                            &physarum_buffers.uniform_buffer,
                            0,
                            bytemuck::cast_slice(&uniform_data),
                        );
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, deposit_bind_group, &[]);
                        pass.dispatch_workgroups(
                            (constants::NUM_PARTICLES + 127) / 128,
                            1,
                            1,
                        );
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
                        let uniform_data = [
                            constants::WIDTH,
                            constants::HEIGHT,
                            constants::DECAY_FACTOR.to_bits(),
                        ];
                        queue.write_buffer(
                            &physarum_buffers.uniform_buffer,
                            0,
                            bytemuck::cast_slice(&uniform_data),
                        );
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, diffusion_bind_group, &[]);
                        pass.dispatch_workgroups(
                            constants::WIDTH / constants::WORK_GROUP_SIZE,
                            constants::HEIGHT / constants::WORK_GROUP_SIZE,
                            1,
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
