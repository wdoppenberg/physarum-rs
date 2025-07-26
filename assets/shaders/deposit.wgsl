struct Uniforms {
	width: u32,
	height: u32,
	depositFactor: f32,
};
@group(0) @binding(10) var<uniform> uniforms: Uniforms;

struct ParticlesCounter {
	data: array<atomic<u32>>,
};
@group(0) @binding(3) var<storage, read_write> particlesCounter: ParticlesCounter;

@group(0) @binding(0) var trailRead: texture_2d<f32>;
@group(0) @binding(6) var trailSampler: sampler;
@group(0) @binding(1) var trailWrite: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var displayWrite: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let pix = vec2<i32>(global_id.xy);
	let uvPos = vec2<f32>(pix) / vec2<f32>(f32(uniforms.width), f32(uniforms.height));

	let prevColor = textureSampleLevel(trailRead, trailSampler, uvPos, 0.0).x;

	let index = global_id.y * uniforms.width + global_id.x;

	// Atomically load the particle count for the current pixel.
	let count = f32(atomicLoad(&particlesCounter.data[index]));

	// Calculate the amount of deposit to add to the trail map.
	let limit = 100.0;
	let limitedCount = min(count, limit);
	let addedDeposit = sqrt(limitedCount) * uniforms.depositFactor;

	// Update the trail map.
	let val = prevColor + addedDeposit;
	textureStore(trailWrite, pix, vec4<f32>(val, 0.0, 0.0, 0.0));

	// Determine the display color based on the particle count.
	let countColorValue = tanh(pow(count / 10.0, 1.7));
	let col = clamp(vec3<f32>(countColorValue), vec3<f32>(0.0), vec3<f32>(1.0));
	let outputColor = vec4<f32>(col, 1.0);

	// Create the output color
//	var outputColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);
//
//	// Test 1: Simple linear mapping to see if it works at all
//	let simpleLinear = clamp(count / 100.0, 0.0, 1.0);
//	outputColor = vec4<f32>(simpleLinear, simpleLinear, simpleLinear, 1.0);


//	var outputColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);
//
//	// Debug: Show actual count ranges to understand the scale
//	if (count == 0.0) {
//	    outputColor = vec4<f32>(0.0, 0.0, 0.0, 1.0); // Black for no particles
//	} else if (count >= 1.0 && count < 10.0) {
//	    outputColor = vec4<f32>(0.1, 0.0, 0.0, 1.0); // Very dark red
//	} else if (count >= 10.0 && count < 100.0) {
//	    outputColor = vec4<f32>(0.3, 0.0, 0.0, 1.0); // Dark red
//	} else if (count >= 100.0 && count < 1000.0) {
//	    outputColor = vec4<f32>(0.6, 0.0, 0.0, 1.0); // Medium red
//	} else if (count >= 1000.0 && count < 10000.0) {
//	    outputColor = vec4<f32>(0.9, 0.0, 0.0, 1.0); // Bright red
//	} else if (count >= 10000.0) {
//	    outputColor = vec4<f32>(1.0, 1.0, 1.0, 1.0); // White for extremely high counts
//	}

//	// Create the output color
//	var outputColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);
//
//	// First, let's see if ANY pixels have count = 0 (meaning counters are being cleared somewhere)
//	if (count == 0.0) {
//	    outputColor = vec4<f32>(0.0, 1.0, 0.0, 1.0); // GREEN for zero counts
//	} else {
//	    // Show the raw count as a very scaled down value
//	    let scaledCount = count / 50000.0; // Extreme scaling
//	    outputColor = vec4<f32>(scaledCount, 0.0, 0.0, 1.0); // Red intensity based on count
//	}



	// Write the final color to the display texture.
	textureStore(displayWrite, pix, outputColor);
}