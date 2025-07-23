struct Uniforms {
    width: i32,
    height: i32,
    decayFactor: f32,
};

@group(0) @binding(10) var<uniform> uniforms: Uniforms;

// Read-only texture for the previous state of the trail map.
@group(0) @binding(0) var trailRead: texture_2d<f32>;
// Write-only texture for the new state of the trail map.
@group(0) @binding(1) var trailWrite: texture_storage_2d<r16float, write>;

// A helper function to wrap coordinates, ensuring they stay within the texture bounds.
// This handles negative coordinates correctly, which can occur when sampling neighbors.
fn loopedPosition(pos: vec2<i32>) -> vec2<i32> {
    let w = uniforms.width;
    let h = uniforms.height;
    return vec2<i32>((pos.x % w + w) % w, (pos.y % h + h) % h);
}

@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec2<i32>(global_id.xy);

    var colorSum = vec2<f32>(0.0);
    let kernelSize = 1;

    // Iterate over a 3x3 kernel to sample neighboring pixels.
    for (var i = -kernelSize; i <= kernelSize; i = i + 1) {
        for (var j = -kernelSize; j <= kernelSize; j = j + 1) {
            let samplePos = loopedPosition(pos - vec2<i32>(i, j));
            colorSum += textureLoad(trailRead, samplePos, 0).xy;
        }
    }

    // Average the sampled colors.
    let c = colorSum / pow(f32(2 * kernelSize + 1), 2.0);

    // Apply decay factor and write the result.
    let decayed = c.x * uniforms.decayFactor;
    let cOutput = vec4<f32>(decayed, 0.0, 0.0, 0.0);

    textureStore(trailWrite, pos, cOutput);
}