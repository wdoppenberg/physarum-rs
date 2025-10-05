use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy_egui::{EguiContexts, egui};
use crate::simulation::resources::ui::UiState;
use crate::simulation::resources::config::PhysarumConfig;

/// System to handle sidebar UI
pub fn sidebar_ui(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<UiState>,
    mut config: ResMut<PhysarumConfig>,
    diagnostics: Res<DiagnosticsStore>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    // Toggle sidebar with F1 key
    if keyboard_input.just_pressed(KeyCode::F1) {
        ui_state.sidebar_visible = !ui_state.sidebar_visible;
    }

    if !ui_state.sidebar_visible {
        return;
    }

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::SidePanel::left("debug_panel")
        .resizable(true)
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.heading("Debug Info");
            ui.separator();

            // Display FPS
            if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(fps_value) = fps.smoothed() {
                    ui.label(format!("FPS: {:.1}", fps_value));
                }
            }

            // Display frame time
            if let Some(frame_time) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) {
                if let Some(frame_time_value) = frame_time.smoothed() {
                    ui.label(format!("Frame Time: {:.2} ms", frame_time_value));
                }
            }

            ui.separator();
            ui.heading("Physarum Settings");
            ui.separator();

            ui.label("Display Settings:");
            ui.add(egui::Slider::new(&mut config.display_factor, 1..=10)
                .text("Display Factor"));
            ui.add(egui::Slider::new(&mut config.pixel_scale_factor, 0.1..=5.0)
                .text("Pixel Scale"));

            ui.separator();
            ui.label("Simulation Settings:");
            
            ui.add(egui::Slider::new(&mut config.decay_factor, 0.0..=1.0)
                .text("Decay Factor"));
            ui.add(egui::Slider::new(&mut config.deposit_factor, 0.0..=10.0)
                .text("Deposit Factor"));

            ui.separator();
            ui.label("Color Mode:");
            ui.horizontal(|ui| {
                ui.radio_value(&mut config.color_mode, 0, "Mode 0");
                ui.radio_value(&mut config.color_mode, 1, "Mode 1");
                ui.radio_value(&mut config.color_mode, 2, "Mode 2");
            });

            ui.separator();
            ui.label(format!("Particles: {}", config.num_particles));
            ui.label(format!("Resolution: {}x{}", config.width, config.height));

            ui.separator();
            ui.label("Press F1 to toggle this panel");
        });
}
