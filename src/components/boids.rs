use bevy::prelude::*;

#[derive(Component)]
pub(crate) struct Position(Vec2);

#[derive(Component)]
pub(crate) struct Velocity(Vec2);

#[derive(Component)]
pub(crate) struct Mass(f32);

#[derive(Component)]
pub(crate) struct Size(f32);

#[derive(Component)]
pub(crate) enum Behaviour {
    Random {
        /// Perturbation factor
        p: f32,
        /// Bias
        bias: Vec2,
    },
    Target {
        pos: Vec2,
    },
    Flock,
}

/// Interaction with physarum field
#[derive(Component)]
pub(crate) enum Interaction {
    Attract,
    Repel,
    Channel { dir: Vec2 },
}

#[derive(Bundle)]
pub(crate) struct BoidBundle {
    position: Position,
    velocity: Velocity,
    size: Size,
    mass: Mass,
    behaviour: Behaviour,
    interaction: Interaction,
}

impl Default for BoidBundle {
    fn default() -> Self {
        Self {
            position: Position(Vec2::ZERO),
            velocity: Velocity(Vec2::ZERO),
            size: Size(1.0),
            mass: Mass(1.0),
            behaviour: Behaviour::Random {
                p: 0.01,
                bias: Vec2::ZERO,
            },
            interaction: Interaction::Attract,
        }
    }
}
