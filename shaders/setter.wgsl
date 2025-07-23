struct Uniforms {
    width: u32,
    height: u32,
    value: u32,
};

// Uniforms are grouped in a struct.
// The binding group and index can be adjusted to fit your pipeline layout.
@group(0) @binding(10) var<uniform> uniforms: Uniforms;

// This buffer is for particle counters.
// It's declared as an array of atomic integers to allow safe parallel access.
struct ParticlesCounter {
    data: array<atomic<u32>>,
};
@group(0) @binding(3) var<storage, read_write> particlesCounter: ParticlesCounter;

@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Calculate the 1D index from the 2D invocation ID.
    let index = global_id.x * uniforms.height + global_id.y;
    // Reset the counter at the given pixel to the specified value.
    atomicStore(&particlesCounter.data[index], uniforms.value);
}
