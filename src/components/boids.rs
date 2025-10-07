use bevy::prelude::Component;

/// Represents a single boid that applies bias to the simulation
/// Now a component that can be attached to entities
#[derive(Component, Clone, Copy, Debug)]
pub struct Boid {
    /// Position in simulation texture coordinates
    pub x: f32,
    pub y: f32,
    /// Movement bias direction (normalized)
    pub move_bias_x: f32,
    pub move_bias_y: f32,
    /// L2 norm of the bias (length/strength)
    pub l2: f32,
}

impl Default for Boid {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            move_bias_x: 0.0,
            move_bias_y: 0.0,
            l2: 0.0,
        }
    }
}
