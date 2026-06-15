use crate::render::{PhysarumSimulationLabel, PhysarumSimulationNode};
use crate::resources::audio::{AudioAnalysisState, AudioCaptureStatus};
use crate::resources::config::PhysarumConfig;
use crate::resources::input::PhysarumInputState;
use crate::resources::render::PhysarumImages;
use crate::resources::ui::UiState;
use crate::systems::audio::{setup_audio_capture, update_audio_reactivity};
use crate::systems::input::handle_input;
use crate::systems::post_process::update_post_process_settings;
use crate::systems::render::{
    handle_buffer_resize, init_physarum_pipeline, prepare_bind_groups, update_simulation_params,
    RenderWorldDimensions,
};
use crate::systems::resize::handle_window_resize;
use crate::systems::ui::sidebar_ui;
use bevy::app::{App, Plugin, Startup, Update};
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
        app.init_resource::<AudioAnalysisState>();
        app.init_resource::<AudioCaptureStatus>();

        // Initialize the UI state resource
        app.init_resource::<UiState>();

        // Register the pipeline status resource
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<PhysarumConfig>();
        render_app.init_resource::<RenderWorldDimensions>();

        render_app
            .add_systems(RenderStartup, init_physarum_pipeline)
            .add_systems(
                Render,
                (
                    handle_buffer_resize.in_set(RenderSystems::PrepareResources),
                    update_simulation_params.in_set(RenderSystems::PrepareResources),
                    prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            );

        let mut render_graph: Mut<RenderGraph> = render_app.world_mut().resource_mut();
        render_graph.add_node(PhysarumSimulationLabel, PhysarumSimulationNode::default());
        render_graph.add_node_edge(
            PhysarumSimulationLabel,
            bevy::render::graph::CameraDriverLabel,
        );

        app.add_systems(Startup, setup_audio_capture)
            .add_systems(
                Update,
                (
                    handle_window_resize,
                    handle_input,
                    update_audio_reactivity,
                    update_post_process_settings.run_if(resource_changed::<PhysarumConfig>),
                ),
            )
            .add_systems(EguiPrimaryContextPass, sidebar_ui);
    }
}
