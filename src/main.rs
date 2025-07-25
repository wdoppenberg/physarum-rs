use bevy::prelude::*;

mod points_basematrix;
mod simulation;

use simulation::PhysarumPlugin;
use crate::simulation::systems::{setup};

fn main() -> AppExit {
    // Create the app with default plugins and window configuration
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Physarum Simulation".into(),
                        resolution: (
                            simulation::resources::simulation_settings::WIDTH as f32,
                            simulation::resources::simulation_settings::HEIGHT as f32,
                        )
                            .into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            PhysarumPlugin,
        ))
        .add_systems(Startup, setup)

        .run()
}
