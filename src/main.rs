use bevy::prelude::*;

pub mod simulation;

use simulation::plugin::PhysarumPlugin;
use simulation::resources::config::PhysarumConfig;
use simulation::systems::render::render_setup;
use bevy::input_focus::InputDispatchPlugin;

fn main() -> AppExit {
    // Create default config and use it for window setup
    let config = PhysarumConfig::default();
    // Create the app with default plugins and window configuration
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(config.clone())
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Physarum Simulation".into(),
                        resolution: (
                            config.width,
                            config.height,
                        )
                            .into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            // New UI core widgets and input dispatch
            InputDispatchPlugin,
            PhysarumPlugin,
        ))
        .add_systems(Startup, render_setup)
        .run()
}
