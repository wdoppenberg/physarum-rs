const pi = 3.14159265359;

struct Uniforms {
    width: u32,
    height: u32,
    value: f32,          // pixelScaleFactor provided via 'value'
    colorMode: u32,
    num_particles: u32,
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
};
@group(0) @binding(10) var<uniform> uniforms: Uniforms;

struct BoidData {
    x: f32,
    y: f32,
    moveBiasX: f32,
    moveBiasY: f32,
    l2: f32,
    _padding: vec3<f32>,
};

@group(0) @binding(11) var<storage, read> boids: array<BoidData, 50>;

struct SimulationSettings {
    default_scaling_factor: f32,
    sensor_distance0: f32,
    sd_exponent: f32,
    sd_amplitude: f32,
    sensor_angle0: f32,
    sa_exponent: f32,
    sa_amplitude: f32,
    rotation_angle0: f32,
    ra_exponent: f32,
    ra_amplitude: f32,
    move_distance0: f32,
    md_exponent: f32,
    md_amplitude: f32,
    sensor_bias1: f32,
    sensor_bias2: f32,
};

@group(0) @binding(5) var<storage, read> pointParams: array<SimulationSettings>;
@group(0) @binding(0) var trailRead: texture_2d<f32>;
@group(0) @binding(6) var trailSampler: sampler;

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

// Simple 3D value noise (ported)
fn random3(st: vec3<f32>) -> f32 {
    return fract(sin(dot(st, vec3<f32>(12.9898, 78.233, 151.7182))) * 43758.5453123);
}

fn noise3(st: vec3<f32>) -> f32 {
    let i = floor(st);
    let F = fract(st);
    let a = random3(i);
    let b = random3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = random3(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = random3(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = random3(i + vec3<f32>(0.0, 0.0, 1.0));
    let f = random3(i + vec3<f32>(1.0, 0.0, 1.0));
    let g = random3(i + vec3<f32>(0.0, 1.0, 1.0));
    let h = random3(i + vec3<f32>(1.0, 1.0, 1.0));
    let u = F * F * (3.0 - 2.0 * F);
    return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y), mix(mix(e, f, u.x), mix(g, h, u.x), u.y), u.z);
}

fn float_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

// Manual f16 pack/unpack to avoid requiring SHADER_FLOAT16_IN_FLOAT32 capability
fn f32_to_f16_bits(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    let sign = (bits >> 31u) & 1u;
    let exp = (bits >> 23u) & 0xFFu;
    let mantissa = bits & 0x7FFFFFu;
    if (exp == 255u) {
        return (sign << 15u) | 0x7C00u | (mantissa >> 13u);
    }
    if (exp == 0u) {
        return (sign << 15u);
    }
    let new_exp = i32(exp) - 127 + 15;
    if (new_exp >= 31) {
        return (sign << 15u) | 0x7C00u;
    }
    if (new_exp <= 0) {
        return (sign << 15u);
    }
    return (sign << 15u) | (u32(new_exp) << 10u) | (mantissa >> 13u);
}

fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;
    if (exp == 0x1Fu) {
        return bitcast<f32>((sign << 31u) | 0x7F800000u | (mantissa << 13u));
    }
    if (exp == 0u) {
        return bitcast<f32>(sign << 31u);
    }
    return bitcast<f32>((sign << 31u) | ((exp + 127u - 15u) << 23u) | (mantissa << 13u));
}

fn pack2x16float_sw(v: vec2<f32>) -> u32 {
    return f32_to_f16_bits(v.x) | (f32_to_f16_bits(v.y) << 16u);
}

fn unpack2x16float_sw(packed: u32) -> vec2<f32> {
    return vec2<f32>(
        f16_bits_to_f32(packed & 0xFFFFu),
        f16_bits_to_f32((packed >> 16u) & 0xFFFFu)
    );
}

fn getGridValue(pos: vec2<f32>) -> f32 {
    let w = f32(uniforms.width);
    let h = f32(uniforms.height);
    // Add 0.5 for rounding, then wrap coordinates.
    let tex_pos = vec2<f32>(
        float_mod(pos.x, w) / w,
        float_mod(pos.y, h) / h
    );
    return textureSampleLevel(trailRead, trailSampler, tex_pos, 0.0).x;
}

fn senseFromAngle(angle: f32, pos: vec2<f32>, heading: f32, so: f32) -> f32 {
    let sense_pos = pos + vec2<f32>(so * cos(heading + angle), so * sin(heading + angle));
    return getGridValue(sense_pos);
}

const DISPATCH_WIDTH = 65535u;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Calculate particle index from 2D dispatch
    let particle_idx = global_id.y * DISPATCH_WIDTH + global_id.x;

    // Early return if we're beyond the number of particles
    if (particle_idx >= uniforms.num_particles) {
        return;
    }

    // Load particle data
    let pos_idx = 3u * particle_idx;
    let particlePosPacked = particlesArray.data[pos_idx];
    var particlePos = unpack2x16unorm(particlePosPacked) * vec2<f32>(f32(uniforms.width), f32(uniforms.height));

    let curProgressAndHeadingPacked = particlesArray.data[pos_idx + 1u];
    let curProgressAndHeading = unpack2x16unorm(curProgressAndHeadingPacked) * vec2<f32>(1.0, 2.0 * pi);
    var heading = curProgressAndHeading.y;

    var velocity = unpack2x16float_sw(particlesArray.data[pos_idx + 2u]);

    // Parameters
    let p_bg = pointParams[1]; // background
    let p_pen = pointParams[0]; // pen

    let w = f32(uniforms.width);
    let h = f32(uniforms.height);

    // Normalized positions
    let normalizedPosition = vec2<f32>(particlePos.x / w, particlePos.y / h);
    let normalizedActionPosition = vec2<f32>(uniforms.actionX / w, uniforms.actionY / h);

    // Noise helpers
    let noiseScale = 20.0;
    let noiseScale2 = 6.0;
    let positionForNoise1 = vec2<f32>(normalizedPosition.x * (w / h), normalizedPosition.y) * noiseScale;
    let positionForNoise2 = vec2<f32>(normalizedPosition.x * (w / h), normalizedPosition.y) * noiseScale2;

    // Lerp factor around action using gaussian
    var positionFromAction = normalizedPosition - normalizedActionPosition;
    positionFromAction.x = positionFromAction.x * (w / h);


    let distanceNoiseFactor = 0.9 + 0.2 * noise3(vec3<f32>(positionForNoise2.x, positionForNoise2.y, 0.6 * uniforms.time));
    let distanceFromAction = length(positionFromAction) * distanceNoiseFactor;
    let lerper = exp(-distanceFromAction * distanceFromAction / max(1e-6, uniforms.actionAreaSizeSigma) / max(1e-6, uniforms.actionAreaSizeSigma));

    // Wave effect disabled for now (requires arrays); keep as zero
    let waveSum = 0.0;

    // Sensed value with bias and scaling
    let direction = vec2<f32>(cos(heading), sin(heading));

    let tunedSensorScaler_mix = mix(p_bg.default_scaling_factor, p_pen.default_scaling_factor, lerper) * (1.0 + 0.3 * waveSum);
    let sensorBias1_mix = mix(p_bg.sensor_bias1, p_pen.sensor_bias1, lerper);
    let sensorBias2_mix = mix(p_bg.sensor_bias2, p_pen.sensor_bias2, lerper);

    var currentSensedValue = getGridValue(particlePos + sensorBias2_mix * direction + vec2<f32>(0.0, sensorBias1_mix)) * tunedSensorScaler_mix;
    currentSensedValue = clamp(currentSensedValue, 1e-9, 1.0);

    // Mix parameters
    let SensorDistance0_mix = mix(p_bg.sensor_distance0, p_pen.sensor_distance0, lerper);
    let SD_amplitude_mix = mix(p_bg.sd_amplitude, p_pen.sd_amplitude, lerper);
    let SD_exponent_mix = mix(p_bg.sd_exponent, p_pen.sd_exponent, lerper);

    let MoveDistance0_mix = mix(p_bg.move_distance0, p_pen.move_distance0, lerper);
    let MD_amplitude_mix = mix(p_bg.md_amplitude, p_pen.md_amplitude, lerper);
    let MD_exponent_mix = mix(p_bg.md_exponent, p_pen.md_exponent, lerper);

    let SensorAngle0_mix = mix(p_bg.sensor_angle0, p_pen.sensor_angle0, lerper);
    let SA_amplitude_mix = mix(p_bg.sa_amplitude, p_pen.sa_amplitude, lerper);
    let SA_exponent_mix = mix(p_bg.sa_exponent, p_pen.sa_exponent, lerper);

    let RotationAngle0_mix = mix(p_bg.rotation_angle0, p_pen.rotation_angle0, lerper);
    let RA_amplitude_mix = mix(p_bg.ra_amplitude, p_pen.ra_amplitude, lerper);
    let RA_exponent_mix = mix(p_bg.ra_exponent, p_pen.ra_exponent, lerper);

    let sensorDistance = SensorDistance0_mix + SD_amplitude_mix * pow(currentSensedValue, SD_exponent_mix) * uniforms.value;
    let moveDistance = MoveDistance0_mix + MD_amplitude_mix * pow(currentSensedValue, MD_exponent_mix) * uniforms.value;
    let sensorAngle = SensorAngle0_mix + SA_amplitude_mix * pow(currentSensedValue, SA_exponent_mix);
    let rotationAngle = RotationAngle0_mix + RA_amplitude_mix * pow(currentSensedValue, RA_exponent_mix);

    // Sensing 3 directions
    let sensedLeft = senseFromAngle(-sensorAngle, particlePos, heading, sensorDistance);
    let sensedMiddle = senseFromAngle(0.0, particlePos, heading, sensorDistance);
    let sensedRight = senseFromAngle(sensorAngle, particlePos, heading, sensorDistance);

    var newHeading = heading;
    if (sensedMiddle > sensedLeft && sensedMiddle > sensedRight) {
        // keep
    } else if (sensedMiddle < sensedLeft && sensedMiddle < sensedRight) {
        newHeading = select(heading + rotationAngle, heading - rotationAngle, random01FromParticle(particlePos) < 0.5);
    } else if (sensedRight < sensedLeft) {
        newHeading = heading - rotationAngle;
    } else if (sensedLeft < sensedRight) {
        newHeading = heading + rotationAngle;
    }

    // Move bias from action and noise (mouse/keyboard)
    let noiseValue = noise3(vec3<f32>(positionForNoise1.x, positionForNoise1.y, 0.8 * uniforms.time));
    let moveBiasFactor = 5.0 * lerper * noiseValue;
    var moveBias = moveBiasFactor * vec2<f32>(uniforms.moveBiasActionX, uniforms.moveBiasActionY);

    // Add bias from all active boids
    for (var i: u32 = 0u; i < uniforms.numBoids; i = i + 1u) {
        let boid = boids[i];
        let boidPos = vec2<f32>(boid.x, boid.y);
        let normalizedBoidPos = vec2<f32>(boid.x / w, boid.y / h);

        // Calculate distance from particle to boid (similar to action point)
        var positionFromBoid = normalizedPosition - normalizedBoidPos;
        positionFromBoid.x = positionFromBoid.x * (w / h);

        let distanceFromBoid = length(positionFromBoid) * distanceNoiseFactor;
        let boidLerper = exp(-distanceFromBoid * distanceFromBoid / max(1e-6, uniforms.actionAreaSizeSigma) / max(1e-6, uniforms.actionAreaSizeSigma));

        // Apply boid's bias with distance-based falloff
        let boidBiasFactor = 5.0 * boidLerper * noiseValue;
        moveBias = moveBias + boidBiasFactor * vec2<f32>(boid.moveBiasX, boid.moveBiasY);
    }

    // Classic position
    let classicNewPosition = particlePos + vec2<f32>(moveDistance * cos(newHeading), moveDistance * sin(newHeading)) + moveBias;

    // Inertia
    velocity = velocity * 0.98;
    let vf = 1.0;
    let velocityBias = 0.2 * uniforms.L2Action;
    let vx = velocity.x + vf * cos(newHeading) + velocityBias * moveBias.x;
    let vy = velocity.y + vf * sin(newHeading) + velocityBias * moveBias.y;

    let dt = 0.07 * pow(moveDistance, 1.4);
    let inertiaNewPosition = particlePos + dt * vec2<f32>(vx, vy) + moveBias;

    let moveStyleLerper = 0.6 * uniforms.L2Action + 0.8 * waveSum;
    var nextPos = mix(classicNewPosition, inertiaNewPosition, moveStyleLerper);

    // Spawning (limited: circular spawn only when spawnParticles == 1)
    var shouldSpawn = false;
    var spawnCenterX = uniforms.actionX;
    var spawnCenterY = uniforms.actionY;
    
    // Check if spawning from action point (F key pressed)
    if (uniforms.spawnParticles >= 1u) {
        shouldSpawn = true;
    }

    // Check if spawning from any boid
    for (var i: u32 = 0u; i < uniforms.numBoids; i = i + 1u) {
        let boid = boids[i];
        let boidPos = vec2<f32>(boid.x, boid.y);
        let normalizedBoidPos = vec2<f32>(boid.x / w, boid.y / h);

        // Calculate distance from particle to boid
        var positionFromBoid = normalizedPosition - normalizedBoidPos;
        positionFromBoid.x = positionFromBoid.x * (w / h);

        let distanceFromBoid = length(positionFromBoid) * distanceNoiseFactor;
        let boidLerper = exp(-distanceFromBoid * distanceFromBoid / max(1e-6, uniforms.actionAreaSizeSigma) / max(1e-6, uniforms.actionAreaSizeSigma));

        // If particle is close enough to this boid, spawn from boid
        if (boidLerper > 0.1) {
            shouldSpawn = true;
            spawnCenterX = boid.x;
            spawnCenterY = boid.y;
            break; // Use first matching boid
        }
    }
    
    if (shouldSpawn) {
        let randForChoice = random01FromParticle(particlePos * 1.1);
        if (randForChoice < uniforms.spawnFraction) {
            let randForRadius = random01FromParticle(particlePos * 2.2);
            // circular spawn
            let randForTheta = random01FromParticle(particlePos * 3.3);
            let theta = randForTheta * pi * 2.0;
            let r1 = uniforms.actionAreaSizeSigma * 0.55 * (0.95 + 0.1 * randForRadius);
            let spos = r1 * vec2<f32>(cos(theta), sin(theta));
            let spos_px = spos * h;
            nextPos = vec2<f32>(spawnCenterX + spos_px.x, spawnCenterY + spos_px.y);
        }
    }

    // Wrap positions
    nextPos = vec2<f32>(float_mod(nextPos.x + w, w), float_mod(nextPos.y + h, h));

    // Deposit counter
    let xi = u32(round(clamp(nextPos.x, 0.0, w - 1.0)));
    let yi = u32(round(clamp(nextPos.y, 0.0, h - 1.0)));
    let counter_idx = yi * uniforms.width + xi;
    atomicAdd(&particlesCounter.data[counter_idx], 1u);

    // Respawn
    let reinitSegment = 0.0010;
    let curA = curProgressAndHeading.x;
    if (curA < reinitSegment) {
        nextPos = randomPosFromParticle(particlePos);
    }
    let nextA = fract(curA + reinitSegment);

    let nextPosUV = nextPos / vec2<f32>(w, h);
    let newHeadingNorm = fract(newHeading / (2.0 * pi));
    let nextAandHeading = vec2<f32>(nextA, newHeadingNorm);

    // Store back
    particlesArray.data[pos_idx] = pack2x16unorm(nextPosUV);
    particlesArray.data[pos_idx + 1u] = pack2x16unorm(nextAandHeading);
    particlesArray.data[pos_idx + 2u] = pack2x16float_sw(vec2<f32>(vx, vy));
}
