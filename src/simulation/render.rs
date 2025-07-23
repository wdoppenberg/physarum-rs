use bevy::prelude::*;
use bevy::render::render_resource::*;
use std::borrow::Cow;
use std::thread::sleep;
use std::time::Duration;
use super::resources::PipelineStatus;

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
        entry_point: Cow::Owned(entry_point.to_string()),
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

/// System to check if all pipelines are ready
pub fn check_pipeline_status(
    mut pipeline_status: ResMut<PipelineStatus>,
    pipeline_cache: Option<Res<PipelineCache>>,
    simulation: Option<Res<super::resources::PhysarumSimulation>>,
) {
    // Log that the system is running
    info!("Checking pipeline status...");
    
    // If simulation resource doesn't exist yet, we can't check pipeline status
    let simulation = match simulation {
        Some(sim) => sim,
        None => {
            info!("Simulation resource not available yet");
            return;
        }
    };
    
    // If PipelineCache is not available, we can't check pipeline status
    let pipeline_cache = match pipeline_cache {
        Some(cache) => cache,
        None => {
            info!("PipelineCache not available yet");
            sleep(Duration::from_secs(1));
            return;
        }
    };
    
    // If pipeline IDs are invalid, we can't check pipeline status
    if simulation.setter_pipeline_id == CachedComputePipelineId::INVALID ||
       simulation.move_pipeline_id == CachedComputePipelineId::INVALID ||
       simulation.deposit_pipeline_id == CachedComputePipelineId::INVALID ||
       simulation.diffusion_pipeline_id == CachedComputePipelineId::INVALID {
        info!("Pipeline IDs not initialized yet");
        return;
    }
    
    info!("Checking pipeline status for IDs: setter={:?}, move={:?}, deposit={:?}, diffusion={:?}",
          simulation.setter_pipeline_id,
          simulation.move_pipeline_id,
          simulation.deposit_pipeline_id,
          simulation.diffusion_pipeline_id);

    // Check each pipeline
    pipeline_status.setter_ready = check_pipeline_ready(&pipeline_cache, simulation.setter_pipeline_id);
    pipeline_status.move_ready = check_pipeline_ready(&pipeline_cache, simulation.move_pipeline_id);
    pipeline_status.deposit_ready = check_pipeline_ready(&pipeline_cache, simulation.deposit_pipeline_id);
    pipeline_status.diffusion_ready = check_pipeline_ready(&pipeline_cache, simulation.diffusion_pipeline_id);
    
    // Update all_ready flag
    pipeline_status.all_ready = 
        pipeline_status.setter_ready && 
        pipeline_status.move_ready && 
        pipeline_status.deposit_ready && 
        pipeline_status.diffusion_ready;
    
    // Log status
    if pipeline_status.all_ready {
        info!("All compute pipelines are ready");
    } else {
        // Log which pipelines are still pending
        if !pipeline_status.setter_ready {
            debug!("Waiting for setter pipeline to compile...");
        }
        if !pipeline_status.move_ready {
            debug!("Waiting for move pipeline to compile...");
        }
        if !pipeline_status.deposit_ready {
            debug!("Waiting for deposit pipeline to compile...");
        }
        if !pipeline_status.diffusion_ready {
            debug!("Waiting for diffusion pipeline to compile...");
        }
    }
}