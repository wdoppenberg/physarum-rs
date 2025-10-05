use crate::simulation::render::{PhysarumSimulationLabel, PhysarumSimulationNode};
use crate::simulation::resources::main::PhysarumInputState;
use crate::simulation::resources::render::PhysarumImages;
use crate::simulation::resources::ui::UiState;
use crate::simulation::systems::main::handle_input;
use crate::simulation::systems::ui::sidebar_ui;
use crate::simulation::systems::render::{
    init_physarum_pipeline, prepare_bind_groups, update_simulation_params,
};
use bevy::app::{App, Plugin, Update};
use bevy::prelude::{IntoScheduleConfigs, Mut};
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_graph::RenderGraph;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use crate::simulation::resources::config::PhysarumConfig;

/// Plugin for the Physarum simulation
pub struct PhysarumPlugin;

impl Plugin for PhysarumPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<PhysarumImages>::default(),
            ExtractResourcePlugin::<PhysarumInputState>::default(),
            ExtractResourcePlugin::<PhysarumConfig>::default(),
        ));

        // Initialize the input state resource
        app.init_resource::<PhysarumInputState>();
        
        // Initialize the UI state resource
        app.init_resource::<UiState>();

        // Register the pipeline status resource
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<PhysarumConfig>();

        render_app
            .add_systems(RenderStartup, init_physarum_pipeline)
            .add_systems(
                Render,
                (
                    prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    update_simulation_params.in_set(RenderSystems::Queue),
                ),
            );

        let mut render_graph: Mut<RenderGraph> = render_app.world_mut().resource_mut();
        render_graph.add_node(PhysarumSimulationLabel, PhysarumSimulationNode::default());
        render_graph.add_node_edge(
            PhysarumSimulationLabel,
            bevy::render::graph::CameraDriverLabel,
        );

        app.add_systems(Update, (handle_input, sidebar_ui));
    }
}
