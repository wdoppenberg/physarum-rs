use bevy::log::debug;
use crate::components::boids::Boid;
use crate::resources::config::PhysarumConfig;
use bevy::prelude::{Commands, Query, Res, Time};
use bevy::math::Vec2;

/// Update boid positions and movement
/// This is a simple system that makes boids move around in the simulation space
pub fn update_boids(
    mut boids: Query<&mut Boid>,
    config: Res<PhysarumConfig>,
    time: Res<Time>,
) {
    let sim_width = config.width as f32;
    let sim_height = config.height as f32;
    let delta = time.delta_secs();
    
    // Update each boid entity
    for mut boid in boids.iter_mut() {
        // Simple movement: move in the direction of the bias
        let speed = 5.0; // pixels per second
        boid.x += boid.move_bias_x * speed * delta;
        boid.y += boid.move_bias_y * speed * delta;
        
        // Wrap around screen edges
        if boid.x < 0.0 {
            boid.x += sim_width;
        } else if boid.x >= sim_width {
            boid.x -= sim_width;
        }
        
        if boid.y < 0.0 {
            boid.y += sim_height;
        } else if boid.y >= sim_height {
            boid.y -= sim_height;
        }
    }
}

/// Helper function to spawn a boid entity at a specific position with a given direction
pub fn spawn_boid_at(
    commands: &mut Commands,
    x: f32,
    y: f32,
    direction: Vec2,
) {
    let normalized = if direction.length_squared() > 0.0 {
        direction.normalize()
    } else {
        Vec2::new(1.0, 0.0)
    };
    
    let boid = Boid {
        x,
        y,
        move_bias_x: normalized.x,
        move_bias_y: normalized.y,
        l2: normalized.length(),
    };
    
    commands.spawn(boid);
    debug!("Spawned boid at ({x}, {y}) with direction ({}, {})", normalized.x, normalized.y);
}

/// Example system to spawn random boids for testing
/// This can be triggered by specific key presses or other events
pub fn spawn_random_boid(
    mut commands: Commands,
    config: Res<PhysarumConfig>,
    time: Res<Time>,
) {
    // Spawn a new boid every 2 seconds (for testing) up to 50 max
    if (time.elapsed_secs() % 2.0) < 0.016 {
        let angle = time.elapsed_secs();
        let direction = Vec2::new(angle.cos(), angle.sin());

        let x = rand::random::<f32>() * config.width as f32;
        let y = rand::random::<f32>() * config.height as f32;

        spawn_boid_at(&mut commands, x, y, direction);
    }
}
