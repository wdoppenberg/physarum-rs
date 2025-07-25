pub mod components;
pub mod render;
pub mod resources;
pub mod systems;
pub mod utils;

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_graph::RenderGraph;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use render::*;
use systems::*;
use crate::simulation::resources::PhysarumImages;

/// Plugin for the Physarum simulation
pub struct PhysarumPlugin;

impl Plugin for PhysarumPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(ExtractResourcePlugin::<PhysarumImages>::default());
        // Register the pipeline status resource
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .add_systems(RenderStartup, init_physarum_pipeline)
            .add_systems(
                Render,
                prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph: Mut<RenderGraph> = render_app.world_mut().resource_mut();
        render_graph.add_node(PhysarumSimulationLabel, PhysarumSimulationNode::default());
        render_graph.add_node_edge(
            PhysarumSimulationLabel,
            bevy::render::graph::CameraDriverLabel,
        );
    }
}
