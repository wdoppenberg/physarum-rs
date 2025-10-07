mod buffers;
pub mod constants;
pub mod plugin;
pub mod render;
pub mod resources;
pub mod systems;
pub mod utils;
mod components;

use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;

use bevy::input_focus::InputDispatchPlugin;
use bevy_egui::EguiPlugin;
use plugin::PhysarumPlugin;
use resources::config::PhysarumConfig;
use systems::render::render_setup;

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
                        resolution: (config.width, config.height).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            // New UI core widgets and input dispatch
            InputDispatchPlugin,
            EguiPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            PhysarumPlugin,
        ))
        .add_systems(Startup, render_setup)
        .run()
}
