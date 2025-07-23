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
  @group(0) @binding(0) var trailSampler: sampler;
  @group(0) @binding(1) var trailWrite: texture_storage_2d<r32float, write>;
  @group(0) @binding(4) var displayWrite: texture_storage_2d<rgba8unorm, write>;

  @compute @workgroup_size(32, 32, 1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let pix = vec2<i32>(global_id.xy);
      let uvPos = vec2<f32>(pix) / vec2<f32>(uniforms.width, uniforms.height);

      let prevColor = textureSample(trailRead, trailSampler, uvPos).xy;

      let index = global_id.x * uniforms.height + global_id.y;
      // Atomically load the particle count for the current pixel.
      let count = f32(atomicLoad(&particlesCounter.data[index]));

      // Calculate the amount of deposit to add to the trail map.
      let LIMIT = 100.0;
      let limitedCount = min(count, LIMIT);
      let addedDeposit = sqrt(limitedCount) * uniforms.depositFactor;

      // Update the trail map.
      let val = prevColor.x + addedDeposit;
      textureStore(trailWrite, pix, vec4<f32>(val, 0.0, 0.0, 0.0));

      // Determine the display color based on the particle count.
      let countColorValue = tanh(pow(count / 10.0, 1.7));
      let col = clamp(vec3<f32>(countColorValue), vec3<f32>(0.0), vec3<f32>(1.0));
      let outputColor = vec4<f32>(col, 1.0);

      // Write the final color to the display texture.
      textureStore(displayWrite, pix, outputColor);
  }