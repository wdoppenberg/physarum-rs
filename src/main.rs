use bevy::prelude::*;

pub mod simulation;
pub mod ui;

use simulation::plugin::PhysarumPlugin;
use simulation::systems::render::render_setup;
use ui::UiPlugin;
use bevy::core_widgets::CoreWidgetsPlugins;
use bevy::input_focus::InputDispatchPlugin;

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
            // New UI core widgets and input dispatch
            CoreWidgetsPlugins,
            InputDispatchPlugin,
            // Simulation + our UI plugin
            PhysarumPlugin,
            // UiPlugin,
        ))
        .add_systems(Startup, render_setup)
        .run()
}
