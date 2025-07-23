pub mod components;
pub mod resources;
pub mod systems;
pub mod utils;
pub mod render;

use bevy::prelude::*;

use systems::*;
use render::*;
use resources::PipelineStatus;

/// Plugin for the Physarum simulation
pub struct PhysarumPlugin;

impl Plugin for PhysarumPlugin {
    fn build(&self, app: &mut App) {
        // Register the pipeline status resource
        app.init_resource::<PipelineStatus>();

        // Add systems to setup the simulation
        app.add_systems(PostStartup, (
            setup_resources,
            setup_display,
        ).chain());

        // Add system to check pipeline status
        app.add_systems(Update, check_pipeline_status);

        // Add systems to update the simulation
        app.add_systems(Update, (
            handle_input,
            update_simulation,
        ));
    }
}