const pi = 3.14159265359;

struct Uniforms {
    width: u32,
    height: u32,
    pixelScaleFactor: f32,
};
@group(0) @binding(10) var<uniform> uniforms: Uniforms;

struct PointSettings {
    defaultScalingFactor: f32,
    SensorDistance0: f32,
    SD_exponent: f32,
    SD_amplitude: f32,
    SensorAngle0: f32,
    SA_exponent: f32,
    SA_amplitude: f32,
    RotationAngle0: f32,
    RA_exponent: f32,
    RA_amplitude: f32,
    MoveDistance0: f32,
    MD_exponent: f32,
    MD_amplitude: f32,
    SensorBias1: f32,
    SensorBias2: f32,
};

@group(0) @binding(5) var<storage, read> pointParams: array<PointSettings>;
@group(0) @binding(0) var trailRead: texture_2d<f32>;
@group(0) @binding(0) var trailSampler: sampler;

struct ParticlesCounter {
    data: array<atomic<u32>>,
};
@group(0) @binding(3) var<storage, read_write> particlesCounter: ParticlesCounter;

struct Particles {
    data: array<u32>,
};
@group(0) @binding(2) var<storage, read_write> particlesArray: Particles;

// --- Randomness utilities (PCG hash) ---
fn pcg_hash(v_in: u32) -> u32 {
    var v = v_in * 747796405u + 2891336453u;
    let word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randFloat(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) / 4294967296.0;
}

fn randomPosFromParticle(particlePos: vec2<f32>) -> vec2<f32> {
    let ipos = vec2<u32>(floor(particlePos));
    var seed = (ipos.x & 0xFFFFu) | ((ipos.y & 0xFFFFu) << 16u);
    let rx = randFloat(&seed);
    let ry = randFloat(&seed);
    return vec2<f32>(rx * f32(uniforms.width), ry * f32(uniforms.height));
}

fn random01FromParticle(particlePos: vec2<f32>) -> f32 {
    let ipos = vec2<u32>(floor(particlePos));
    var seed = (ipos.x & 0xFFFFu) | ((ipos.y & 0xFFFFu) << 16u);
    return randFloat(&seed);
}

// --- Simulation Logic ---

fn float_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn getGridValue(pos: vec2<f32>) -> f32 {
    let w = f32(uniforms.width);
    let h = f32(uniforms.height);
    // Add 0.5 for rounding, then wrap coordinates.
    let tex_pos = vec2<f32>(
        float_mod(pos.x, w) / w,
        float_mod(pos.y, h) / h
    );
    return textureSample(trailRead, trailSampler, tex_pos).x;
}

fn senseFromAngle(angle: f32, pos: vec2<f32>, heading: f32, so: f32) -> f32 {
    let sense_pos = pos + vec2<f32>(so * cos(heading + angle), so * sin(heading + angle));
    return getGridValue(sense_pos);
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let params = pointParams[0];
    let particle_idx = global_id.x;

    let particlePosPacked = particlesArray.data[2u * particle_idx];
    var particlePos = unpack2x16unorm(particlePosPacked) * vec2<f32>(f32(uniforms.width), f32(uniforms.height));

    let curProgressAndHeadingPacked = particlesArray.data[2u * particle_idx + 1u];
    let curProgressAndHeading = unpack2x16unorm(curProgressAndHeadingPacked) * vec2<f32>(1.0, 2.0 * pi);
    var heading = curProgressAndHeading.y;

    let direction = vec2<f32>(cos(heading), sin(heading));

    var currentSensedValue = getGridValue(particlePos + params.SensorBias2 * direction + vec2<f32>(0.0, params.SensorBias1));
    currentSensedValue = clamp(currentSensedValue * params.defaultScalingFactor, 0.000000001, 1.0);

    let sensorDistance = params.SensorDistance0 + params.SD_amplitude * pow(currentSensedValue, params.SD_exponent) * uniforms.pixelScaleFactor;
    let moveDistance = params.MoveDistance0 + params.MD_amplitude * pow(currentSensedValue, params.MD_exponent) * uniforms.pixelScaleFactor;
    let sensorAngle = params.SensorAngle0 + params.SA_amplitude * pow(currentSensedValue, params.SA_exponent);
    let rotationAngle = params.RotationAngle0 + params.RA_amplitude * pow(currentSensedValue, params.RA_exponent);

    let sensedLeft = senseFromAngle(-sensorAngle, particlePos, heading, sensorDistance);
    let sensedMiddle = senseFromAngle(0.0, particlePos, heading, sensorDistance);
    let sensedRight = senseFromAngle(sensorAngle, particlePos, heading, sensorDistance);

    var newHeading = heading;
    if (sensedMiddle > sensedLeft && sensedMiddle > sensedRight) {
        // Continue straight
    } else if (sensedMiddle < sensedLeft && sensedMiddle < sensedRight) {
        newHeading += rotationAngle * (2.0 * step(0.5, random01FromParticle(particlePos)) - 1.0);
    } else if (sensedRight < sensedLeft) {
        newHeading -= rotationAngle;
    } else if (sensedLeft < sensedRight) {
        newHeading += rotationAngle;
    }

    let px = particlePos.x + moveDistance * cos(newHeading);
    let py = particlePos.y + moveDistance * sin(newHeading);

    let w = f32(uniforms.width);
    let h = f32(uniforms.height);
    var nextPos = vec2<f32>(float_mod(px, w), float_mod(py, h));

    let counter_idx = u32(floor(nextPos.x)) * uniforms.height + u32(floor(nextPos.y));
    atomicAdd(&particlesCounter.data[counter_idx], 1u);

    let reinitSegment = 0.0010;
    let curProgress = curProgressAndHeading.x;
    if (curProgress < reinitSegment) {
        nextPos = randomPosFromParticle(particlePos);
    }

    let nextA = fract(curProgress + reinitSegment);

    let nextPosUV = nextPos / vec2<f32>(w, h);
    let newHeadingNorm = fract(newHeading / (2.0 * pi));
    let nextAandHeading = vec2<f32>(nextA, newHeadingNorm);

    particlesArray.data[2u * particle_idx] = pack2x16unorm(nextPosUV);
    particlesArray.data[2u * particle_idx + 1u] = pack2x16unorm(nextAandHeading);
}