use crate::render::{PhysarumSimulationLabel, PhysarumSimulationNode};
use crate::resources::config::PhysarumConfig;
use crate::resources::input::PhysarumInputState;
use crate::resources::render::PhysarumImages;
use crate::resources::ui::UiState;
use crate::systems::main::handle_input;
use crate::systems::post_process::update_post_process_settings;
use crate::systems::render::{
    init_physarum_pipeline, prepare_bind_groups, update_simulation_params,
};
use crate::systems::ui::sidebar_ui;
use bevy::app::{App, Plugin, Update};
use bevy::prelude::{resource_changed, IntoScheduleConfigs, Mut};
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_graph::RenderGraph;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy_egui::EguiPrimaryContextPass;

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

        app.add_systems(
            Update,
            (
                handle_input,
                update_post_process_settings.run_if(resource_changed::<PhysarumConfig>),
            ),
        )
        .add_systems(EguiPrimaryContextPass, sidebar_ui);
    }
}
