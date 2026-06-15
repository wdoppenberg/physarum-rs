use crate::components::boids::{BoidBundle, Interaction, Position, Size};
use bevy::prelude::*;

pub(crate) fn spawn_boid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        BoidBundle::default(),
        Mesh2d(meshes.add(Circle::new(10.0))),
        MeshMaterial2d(materials.add(Color::from(LinearRgba::BLUE))),
    ));
}

pub(crate) fn handle_physarum_interactions(interactors: Query<(&Interaction, &Position, &Size)>) {
    let buffer_data = interactors.iter().collect::<Vec<_>>();

    for (i, p, s) in buffer_data {
        match i {
            Interaction::Attract => {}
            Interaction::Repel => {}
            Interaction::Channel { .. } => todo!(),
        }
    }
}
