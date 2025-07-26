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

// Color scheme selector (0-5)
// 0: Original grayscale
// 1: Color gradient based on density
// 2: Multi-channel trail system
// 3: Heat map style
// 4: Organic fungus colors
// 5: Position-based animation
const COLOR_SCHEME: u32 = 1u;

// Heat map color function
fn heatMapColor(value: f32) -> vec3<f32> {
    let v = clamp(value, 0.0, 1.0);
    
    if (v < 0.25) {
        return mix(vec3<f32>(0.0, 0.0, 0.5), vec3<f32>(0.0, 0.5, 1.0), v * 4.0);
    } else if (v < 0.5) {
        return mix(vec3<f32>(0.0, 0.5, 1.0), vec3<f32>(0.0, 1.0, 0.5), (v - 0.25) * 4.0);
    } else if (v < 0.75) {
        return mix(vec3<f32>(0.0, 1.0, 0.5), vec3<f32>(1.0, 1.0, 0.0), (v - 0.5) * 4.0);
    } else {
        return mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), (v - 0.75) * 4.0);
    }
}

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
	
	// Choose color scheme based on the constant
	var col: vec3<f32>;
	
	switch(COLOR_SCHEME) {
		case 0u: {
			// Original grayscale
			col = clamp(vec3<f32>(countColorValue), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 1u: {
			// Color gradient based on density
			col = mix(
				vec3<f32>(0.0, 0.0, 0.0),  // Black for low density
				mix(
					vec3<f32>(0.8, 0.2, 0.1),  // Orange-red for medium density
					vec3<f32>(1.0, 1.0, 0.3),  // Bright yellow for high density
					smoothstep(0.3, 0.8, countColorValue)
				),
				smoothstep(0.0, 0.5, countColorValue)
			);
		}
		case 2u: {
			// Multi-channel trail system (simulate with different factors)
			let redValue = countColorValue;
			let greenValue = countColorValue * 0.7;
			let blueValue = countColorValue * 0.4;
			col = clamp(vec3<f32>(redValue, greenValue, blueValue), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 3u: {
			// Heat map style coloring
			col = heatMapColor(countColorValue);
		}
		case 4u: {
			// Organic fungus colors
			// Base mycelium color (pale cream/white)
			let baseColor = vec3<f32>(0.9, 0.85, 0.7);
			// Growth areas (reddish-brown)
			let growthColor = vec3<f32>(0.6, 0.3, 0.2);
			// Dense areas (dark brown/purple)
			let denseColor = vec3<f32>(0.4, 0.2, 0.3);

			col = mix(
				baseColor,
				mix(growthColor, denseColor, smoothstep(0.4, 0.9, countColorValue)),
				countColorValue
			);
		}
		case 5u: {
			// Position-based color animation (simulating time with position)
			// Using uvPos coordinates to simulate time variation
			let pseudoTime = uvPos.x + uvPos.y;
			
			col = vec3<f32>(
				countColorValue * (0.5 + 0.5 * sin(pseudoTime * 15.0)),
				countColorValue * (0.5 + 0.5 * sin(pseudoTime * 15.0 + 2.0)),
				countColorValue * (0.5 + 0.5 * sin(pseudoTime * 15.0 + 4.0))
			);
		}
		default: {
			// Fallback to grayscale
			col = clamp(vec3<f32>(countColorValue), vec3<f32>(0.0), vec3<f32>(1.0));
		}
	}
	
	let outputColor = vec4<f32>(col, 1.0);

	// Write the final color to the display texture.
	textureStore(displayWrite, pix, outputColor);
}