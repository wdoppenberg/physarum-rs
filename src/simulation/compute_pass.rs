use bevy::asset::Handle;
use bevy::prelude::Shader;
use bevy::render::render_resource;
use bevy::render::render_resource::{BindGroup, BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache};

pub struct WorkGroupSize {
    x: u32,
    y: u32,
    z: u32,
}

impl From<(u32, u32, u32)> for WorkGroupSize {
    fn from(value: (u32, u32, u32)) -> Self {
        Self {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

pub struct ComputePass {
    pub pipeline_id: CachedComputePipelineId,
    pub workgroup_size: WorkGroupSize,
}

impl ComputePass {
    /// Creates a new `ComputePass`.
    pub fn new(
        pipeline_cache: &mut PipelineCache,
        layout: &BindGroupLayout,
        shader: &Handle<Shader>,
        entry_point: impl AsRef<str>,
        workgroup_size: impl Into<WorkGroupSize>,
    ) -> Self {
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: None,
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(entry_point.as_ref().to_owned().into()),
            zero_initialize_workgroup_memory: false,
        };

        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);

        Self {
            pipeline_id,
            workgroup_size: workgroup_size.into(),
        }
    }

	/// Dispatches the compute pass.
    pub fn run<'a>(
        &'a self,
        pass: &mut render_resource::ComputePass<'a>,
        pipeline_cache: &'a PipelineCache,
        bind_group: &'a BindGroup,
    ) {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(self.pipeline_id) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.z);
        }
    }

}
