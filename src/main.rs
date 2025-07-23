use bevy::prelude::*;

mod points_basematrix;
mod simulation;

use simulation::PhysarumPlugin;

fn main() {
    // Create the app with default plugins and window configuration
    let mut app = App::new();

    // Add the default plugins with window configuration
    app.add_plugins(
        DefaultPlugins.set(WindowPlugin {
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
        }),
    );

    // Add our Physarum simulation plugin
    app.add_plugins(PhysarumPlugin);

    // Run the app
    app.run();
}