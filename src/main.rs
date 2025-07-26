use bevy::prelude::*;

pub mod simulation;

use simulation::plugin::PhysarumPlugin;
use simulation::systems::render::render_setup;

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
                            simulation::constants::WIDTH as f32,
                            simulation::constants::HEIGHT as f32,
                        )
                            .into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            PhysarumPlugin,
        ))
        .add_systems(Startup, render_setup)
        .run()
}
