struct Uniforms {
	width: u32,
	height: u32,
	depositFactor: f32,
	colorMode: u32,
	numParticles: u32,
	time: f32,
	actionAreaSizeSigma: f32,
	actionX: f32,
	actionY: f32,
	moveBiasActionX: f32,
	moveBiasActionY: f32,
	L2Action: f32,
	spawnParticles: u32,
	spawnFraction: f32,
	randomSpawnNumber: u32,
	numBoids: u32,
	audioLevel: f32,
	audioBass: f32,
	audioMid: f32,
	audioTreble: f32,
	audioBeat: f32,
	darkProfileEnabled: u32,
	darkMaxLuminance: f32,
	darkContrast: f32,
	darkBlackLift: f32,
	liveliness: f32,
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

/////////////////////////////////////
// Color utility functions

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.b, c.g, K.w, K.z), vec4<f32>(c.g, c.b, K.x, K.y), step(c.b, c.g));
    let q = mix(vec4<f32>(p.x, p.y, p.w, c.r), vec4<f32>(c.r, p.y, p.z, p.x), step(p.x, c.r));

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn changeHue(hsv: vec3<f32>, hueChange: f32) -> vec3<f32> {
    var result = hsv;
    result.x += hueChange;
    result.x = result.x % 1.0;
    return result;
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn pal(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.28318 * (c * t + d));
}

/////////////////////////////////////
// Gradient interpolation functions

fn interpolateGradient5(f: f32, cols: array<vec3<f32>, 5>) -> vec3<f32> {
    let fc = clamp(f, 0.0, 1.0);
    let cur = fc * 4.0;
    let icur = i32(floor(cur));
    let next = min(icur + 1, 4);
    return mix(cols[icur], cols[next], fract(cur));
}

fn interpolateGradient6(f: f32, cols: array<vec3<f32>, 6>) -> vec3<f32> {
    let fc = clamp(f, 0.0, 1.0);
    let cur = fc * 5.0;
    let icur = i32(floor(cur));
    let next = min(icur + 1, 5);
    return mix(cols[icur], cols[next], fract(cur));
}

fn interpolateGradient7(f: f32, cols: array<vec3<f32>, 7>) -> vec3<f32> {
    let fc = clamp(f, 0.0, 1.0);
    let cur = fc * 6.0;
    let icur = i32(floor(cur));
    let next = min(icur + 1, 6);
    return mix(cols[icur], cols[next], fract(cur));
}

// Color palettes
fn gradZorgPurple(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 5>(
        vec3<f32>(0.05, 0.0, 0.15),
        vec3<f32>(0.2, 0.0, 0.4),
        vec3<f32>(0.5, 0.1, 0.8),
        vec3<f32>(1.0, 0.0, 0.9),
        vec3<f32>(0.7, 1.0, 0.3)
    );
    return interpolateGradient5(f, cols);
}

fn gradOrangeBlue(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 7>(
        vec3<f32>(0.1, 0.0, 0.2),
        vec3<f32>(0.0, 0.3, 0.6),
        vec3<f32>(0.0, 0.6, 0.9),
        vec3<f32>(0.2, 0.9, 1.0),
        vec3<f32>(1.0, 0.5, 0.0),
        vec3<f32>(1.0, 0.7, 0.1),
        vec3<f32>(1.0, 1.0, 0.3)
    );
    return interpolateGradient7(f, cols);
}

fn gradGreen(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 7>(
        vec3<f32>(0.0, 0.1, 0.05),
        vec3<f32>(0.0, 0.3, 0.2),
        vec3<f32>(0.0, 0.6, 0.4),
        vec3<f32>(0.2, 1.0, 0.8),
        vec3<f32>(0.5, 1.0, 0.3),
        vec3<f32>(0.9, 1.0, 0.4),
        vec3<f32>(1.0, 1.0, 0.8)
    );
    return interpolateGradient7(f, cols);
}

fn gradTealSunset(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 7>(
        vec3<f32>(0.0, 0.1, 0.2),
        vec3<f32>(0.0, 0.4, 0.5),
        vec3<f32>(0.1, 0.7, 0.8),
        vec3<f32>(0.4, 0.8, 0.9),
        vec3<f32>(0.9, 0.4, 0.6),
        vec3<f32>(1.0, 0.5, 0.3),
        vec3<f32>(1.0, 0.8, 0.2)
    );
    return interpolateGradient7(f, cols);
}

fn gradForestNight(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 7>(
        vec3<f32>(0.0, 0.05, 0.15),
        vec3<f32>(0.0, 0.2, 0.3),
        vec3<f32>(0.0, 0.4, 0.3),
        vec3<f32>(0.1, 0.6, 0.5),
        vec3<f32>(0.2, 0.8, 0.7),
        vec3<f32>(0.5, 0.9, 0.9),
        vec3<f32>(0.9, 1.0, 1.0)
    );
    return interpolateGradient7(f, cols);
}

fn gradPurpleFire(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 7>(
        vec3<f32>(0.1, 0.0, 0.2),
        vec3<f32>(0.3, 0.0, 0.5),
        vec3<f32>(0.5, 0.2, 0.9),
        vec3<f32>(0.8, 0.1, 0.7),
        vec3<f32>(1.0, 0.2, 0.5),
        vec3<f32>(1.0, 0.6, 0.2),
        vec3<f32>(1.0, 1.0, 0.3)
    );
    return interpolateGradient7(f, cols);
}

fn gradArctic(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 7>(
        vec3<f32>(0.0, 0.05, 0.15),
        vec3<f32>(0.0, 0.2, 0.4),
        vec3<f32>(0.0, 0.4, 0.7),
        vec3<f32>(0.1, 0.6, 0.9),
        vec3<f32>(0.3, 0.8, 1.0),
        vec3<f32>(0.6, 0.9, 1.0),
        vec3<f32>(0.9, 1.0, 1.0)
    );
    return interpolateGradient7(f, cols);
}

fn gradCyan(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 7>(
        vec3<f32>(0.0, 0.1, 0.15),
        vec3<f32>(0.0, 0.3, 0.4),
        vec3<f32>(0.0, 0.5, 0.7),
        vec3<f32>(0.1, 0.7, 0.9),
        vec3<f32>(0.3, 0.9, 1.0),
        vec3<f32>(0.5, 1.0, 1.0),
        vec3<f32>(0.8, 1.0, 1.0)
    );
    return interpolateGradient7(f, cols);
}

fn gradNeonInferno(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 6>(
        vec3<f32>(0.1, 0.0, 0.2),
        vec3<f32>(0.4, 0.0, 0.6),
        vec3<f32>(0.8, 0.0, 0.9),
        vec3<f32>(1.0, 0.2, 0.4),
        vec3<f32>(1.0, 0.6, 0.2),
        vec3<f32>(1.0, 1.0, 0.5)
    );
    return interpolateGradient6(f, cols);
}

fn gradSolarDrift(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 7>(
        vec3<f32>(0.2, 0.0, 0.1),
        vec3<f32>(0.5, 0.1, 0.0),
        vec3<f32>(0.8, 0.2, 0.0),
        vec3<f32>(1.0, 0.4, 0.0),
        vec3<f32>(1.0, 0.7, 0.1),
        vec3<f32>(1.0, 0.9, 0.3),
        vec3<f32>(1.0, 1.0, 0.8)
    );
    return interpolateGradient7(f, cols);
}

fn gradPlasmaTwilight(f: f32) -> vec3<f32> {
    let cols = array<vec3<f32>, 5>(
        vec3<f32>(0.1, 0.0, 0.3),
        vec3<f32>(0.2, 0.3, 0.8),
        vec3<f32>(0.4, 0.6, 1.0),
        vec3<f32>(0.8, 0.3, 0.9),
        vec3<f32>(1.0, 0.5, 0.9)
    );
    return interpolateGradient5(f, cols);
}

@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let pix = vec2<i32>(global_id.xy);
	let uvPos = vec2<f32>(pix) / vec2<f32>(f32(uniforms.width), f32(uniforms.height));

	// Read previous trail intensity (single-channel R32 texture)
	let prevVal = textureSampleLevel(trailRead, trailSampler, uvPos, 0.0).x;

	let index = global_id.y * uniforms.width + global_id.x;

	// Get particle count for this pixel
	let count = f32(atomicLoad(&particlesCounter.data[index]));

	// Calculate deposit using the same formula as OpenGL version
	let LIMIT = 100.0;
	let limitedCount = min(count, LIMIT);
	let addedDeposit = sqrt(limitedCount) * uniforms.depositFactor;

	// Update trail map (using .xy components)
	let val = prevVal + addedDeposit;
	// Write back to trail texture (single channel)
	textureStore(trailWrite, pix, vec4<f32>(val, 0.0, 0.0, 0.0));

	// Calculate color intensity from particle count (matching OpenGL formula)
	let countColorValue = pow(tanh(7.5 * pow(max(0.0, (count - 1.0) / 1000.0), 0.3)), 8.5) * 1.1;
	let clampedCountColor = min(1.0, countColorValue);

	// Estimate a local motion/edge metric from gradient magnitude to drive color dynamics
	let texel = vec2<f32>(1.0 / f32(uniforms.width), 1.0 / f32(uniforms.height));
	let sL = textureSampleLevel(trailRead, trailSampler, uvPos - vec2<f32>(texel.x, 0.0), 0.0).x;
	let sR = textureSampleLevel(trailRead, trailSampler, uvPos + vec2<f32>(texel.x, 0.0), 0.0).x;
	let sD = textureSampleLevel(trailRead, trailSampler, uvPos - vec2<f32>(0.0, texel.y), 0.0).x;
	let sU = textureSampleLevel(trailRead, trailSampler, uvPos + vec2<f32>(0.0, texel.y), 0.0).x;
	let dx = sR - sL;
	let dy = sU - sD;
	let grad = clamp(length(vec2<f32>(dx, dy)) * 10.0, 0.0, 1.0);

	// Calculate radial offset for color changes
	let pos = vec2<f32>(pix) - vec2<f32>(f32(uniforms.width) * 0.5, f32(uniforms.height) * 0.5);
	let normalizedPos = pos * (2.0 / f32(uniforms.width + uniforms.height)) * 0.6;
	let offset = length(normalizedPos);

	// Use gradient as a dynamic blender (reacts to edges/motion)
	let blend = smoothstep(0.0, 0.6, grad) * 0.9 + 0.1 * offset;
	let col2 = vec3<f32>(clampedCountColor);

	// Color mode selection
	var col: vec3<f32>;

	switch(uniforms.colorMode) {
		case 0u: { // Rainbow HSV cycling
			let hue = fract(clampedCountColor * 2.0 + offset * 0.5 + grad * 0.3);
			let hsv = vec3<f32>(hue, 0.9, clampedCountColor);
			col = hsv2rgb(hsv);
			col = pow(col, vec3<f32>(0.8)) * 1.3;
		}
		case 1u: { // Psychedelic fire - purple to cyan with motion
			let col1 = gradPurpleFire(tanh(clampedCountColor * 1.5));
			let col3 = gradArctic(tanh(clampedCountColor * 1.5));
			col = mix(col1, col3, blend * 0.8);
			col = clamp(1.6 * pow(col, vec3<f32>(0.9)), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 2u: { // Electric ice with hue shift
			let col1 = gradArctic(fract(tanh(clampedCountColor * 0.8 + offset) + 0.2));
			let hsv = rgb2hsv(col1);
			let shiftedHsv = changeHue(hsv, grad * 0.15);
			col = hsv2rgb(vec3<f32>(shiftedHsv.x, min(shiftedHsv.y * 1.3, 1.0), shiftedHsv.z));
			col = mix(col * 1.4, col2, blend * 0.5);
		}
		case 3u: { // Intense neon inferno
			let col1 = gradPurpleFire(tanh(clampedCountColor * 1.6));
			let col3 = gradNeonInferno(tanh(clampedCountColor * 1.4 + offset * 0.3));
			col = mix(col1, col3, blend);
			col = clamp(1.5 * col, vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 4u: { // Vibrant gold over electric blue
			let col1 = gradOrangeBlue(tanh(clampedCountColor * 1.5 + offset * 0.5));
			let col3 = gradSolarDrift(tanh(clampedCountColor * 1.8));
			col = mix(col1, col3, grad * 0.6);
			col = clamp(1.4 * pow(col, vec3<f32>(0.85)), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 5u: { // Procedural cosmic palette
			let t = clampedCountColor + offset * 0.4 + grad * 0.2;
			col = pal(t, vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 0.33, 0.67));
			col = clamp(col * 1.5, vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 6u: { // Hypersaturated purple dreams
			let col1 = gradZorgPurple(fract(tanh(clampedCountColor * 0.8 + offset * 0.5) + 0.2));
			let col3 = gradPlasmaTwilight(tanh(clampedCountColor * 1.4));
			let mixedCol = mix(col1, col3, blend * 0.7);
			col = clamp(1.8 * pow(mixedCol, vec3<f32>(0.95)), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 7u: { // Neon inferno with arctic contrast
			let col1 = gradNeonInferno(tanh(clampedCountColor * 1.5 + grad * 0.3));
			let col3 = gradArctic(tanh(clampedCountColor * 1.4));
			col = mix(col1, col3, blend * 0.8);
			col = clamp(1.5 * pow(col, vec3<f32>(0.88)), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 8u: { // Explosive yellow-green plasma
			let col1 = gradOrangeBlue(tanh(clampedCountColor * 1.6 + offset * 0.4));
			let colGreen = gradGreen(tanh(clampedCountColor * 2.5 + offset * 0.3));
			let col2_ = mix(vec3<f32>(clamp(1.5 * clampedCountColor, 0.0, 1.0)), colGreen, 0.6);
			let motionMix = 0.7 * blend + 0.3 * smoothstep(0.0, 1.0, grad);
			let col3 = mix(col2_, col1, motionMix);
			let col4 = gradSolarDrift(tanh(clampedCountColor * 1.5 + offset * 0.3));
			var col6 = 1.5 * mix(col4, col3, blend * 0.7);
			col6 = pow(col6, vec3<f32>(1.8));
			col = clamp(max(col6, col3), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 9u: { // Bioluminescent green
			let col1 = gradGreen(tanh(clampedCountColor * 1.6));
			let col3 = gradForestNight(tanh(clampedCountColor * 1.4 + offset * 0.3));
			col = mix(col1, col3, blend * 0.6);
			col = clamp(1.5 * pow(col, vec3<f32>(0.9)), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 10u: { // Procedural rainbow waves
			let t = clampedCountColor * 1.5 + offset * 0.8 + grad * 0.5;
			col = pal(t, vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(1.0, 1.0, 0.5), vec3<f32>(0.8, 0.9, 0.3));
			col = clamp(col * 1.6, vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 11u: { // Teal sunset dreamscape
			let col1 = gradTealSunset(tanh(clampedCountColor * 1.5 + offset * 0.4));
			let col3 = gradPlasmaTwilight(tanh(clampedCountColor * 1.3));
			col = mix(col1, col3, blend * 0.7 + grad * 0.2);
			col = clamp(1.5 * pow(col, vec3<f32>(0.9)), vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 12u: { // Dark projection palette with restrained highlights
			let t = tanh(clampedCountColor * (1.3 + 0.3 * uniforms.audioBass) + offset * 0.25);
			let deepBlue = vec3<f32>(0.01, 0.05, 0.12);
			let coldTeal = vec3<f32>(0.03, 0.25, 0.32);
			let ember = vec3<f32>(0.65, 0.34, 0.16);
			let peak = vec3<f32>(0.96, 0.74, 0.35);
			let base = mix(deepBlue, coldTeal, smoothstep(0.0, 0.55, t));
			let hot = mix(ember, peak, smoothstep(0.35, 1.0, t + blend * 0.2));
			col = mix(base, hot, smoothstep(0.2, 1.0, grad + t * 0.7));
			col *= 0.45 + 0.25 * uniforms.audioLevel + 0.20 * uniforms.audioBeat;
		}
		case 10000u: { // Dynamic cyan-magenta split
			let cyan = vec3<f32>(0.0, 1.0, 1.0) * clampedCountColor * 1.3;
			let magenta = vec3<f32>(1.0, 0.0, 1.0) * grad * 1.5;
			col = clamp(cyan + magenta, vec3<f32>(0.0), vec3<f32>(1.0));
		}
		default: {
			// Fallback to rainbow
			let hue = fract(clampedCountColor * 2.0 + offset * 0.5);
			let hsv = vec3<f32>(hue, 0.8, clampedCountColor);
			col = hsv2rgb(hsv);
		}
	}

	// React to the incoming audio signal before final grading.
	let reactiveBoost = 0.75 + 0.35 * uniforms.audioLevel + 0.25 * uniforms.audioBeat;
	col *= reactiveBoost * clamp(uniforms.liveliness, 0.7, 3.0);

	// Projection-safe grading path to keep rooms dark while preserving contrast.
	if (uniforms.darkProfileEnabled == 1u) {
		let black = vec3<f32>(uniforms.darkBlackLift);
		col = max(col, black);
		col = (col - black) * uniforms.darkContrast + black;
		col = min(col, vec3<f32>(uniforms.darkMaxLuminance));
	}

	col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
	let outputColor = vec4<f32>(col, 1.0);

	textureStore(displayWrite, pix, outputColor);
}
