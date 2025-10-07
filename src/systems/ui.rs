use crate::resources::config::PhysarumConfig;
use crate::resources::ui::UiState;
use crate::utils::load_parameters;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

/// Color mode names matching the shader implementations
const COLOR_MODES: &[(&str, u32)] = &[
    ("Rainbow HSV", 0),
    ("Psychedelic Fire", 1),
    ("Electric Ice", 2),
    ("Neon Inferno", 3),
    ("Gold over Blue", 4),
    ("Cosmic Palette", 5),
    ("Purple Dreams", 6),
    ("Neon Arctic", 7),
    ("Yellow-Green Plasma", 8),
    ("Bioluminescent Green", 9),
    ("Rainbow Waves", 10),
    ("Teal Sunset", 11),
];

/// Simulation mode names from PARAMETERS_MATRIX
const SIMULATION_MODES: &[(&str, usize)] = &[
    ("Pure Multiscale", 0),
    ("Hex Hole Open", 1),
    ("Vertebrata", 2),
    ("Star Network", 3),
    ("Enmeshed Singularities", 4),
    ("Waves Upturn", 5),
    ("More Individuals", 6),
    ("Sloppy Bucky", 7),
    ("Massive Structure", 8),
    ("Speed Modulation", 9),
    ("Transmission Tower", 10),
    ("Ink on White", 11),
    ("Vanishing Points", 12),
    ("Scaling Nodule Emergence", 13),
    ("Hyp Offset", 14),
    ("Strike", 15),
    ("Clear Spaghetti", 16),
    ("Bleuje 1", 17),
    ("Bleuje 2", 18),
    ("Bleuje 3", 19),
    ("Bleuje 4", 20),
    ("Bleuje 5", 21),
    ("Bleuje 6", 22),
    ("Bleuje 7", 23),
];

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
            ui.add(egui::Slider::new(&mut config.display_factor, 1..=10).text("Display Factor"));
            ui.add(
                egui::Slider::new(&mut config.pixel_scale_factor, 0.1..=5.0).text("Pixel Scale"),
            );

            ui.separator();
            ui.label("Simulation Settings:");

            ui.add(egui::Slider::new(&mut config.decay_factor, 0.0..=1.0).text("Decay Factor"));
            ui.add(
                egui::Slider::new(&mut config.deposit_factor, 0.0..=10.0).text("Deposit Factor"),
            );
            ui.add(
                egui::Slider::new(&mut config.action_area_size_sigma, 0.0..=1.0)
                    .text("Action Area Size"),
            );

            ui.separator();
            ui.label("Color Mode:");

            // Find current color mode name
            let current_color_name = COLOR_MODES
                .iter()
                .find(|(_, mode)| *mode == config.color_mode)
                .map(|(name, _)| *name)
                .unwrap_or("Unknown");

            egui::ComboBox::from_label("Color Scheme")
                .selected_text(current_color_name)
                .show_ui(ui, |ui| {
                    for (name, mode) in COLOR_MODES {
                        ui.selectable_value(&mut config.color_mode, *mode, *name);
                    }
                });

            ui.separator();
            ui.label("Simulation Mode:");

            // Find current simulation mode name
            let current_sim_name = SIMULATION_MODES
                .iter()
                .find(|(_, idx)| *idx == config.new_index)
                .map(|(name, _)| *name)
                .unwrap_or("Unknown");

            egui::ComboBox::from_label("Behavior Pattern")
                .selected_text(current_sim_name)
                .show_ui(ui, |ui| {
                    for (name, idx) in SIMULATION_MODES {
                        if ui
                            .selectable_value(&mut config.new_index, *idx, *name)
                            .changed()
                        {
                            // Update the settings when simulation mode changes
                            config.current_settings = load_parameters(config.new_index);
                            config.settings_changed = true;
                        }
                    }
                });

            ui.separator();
            ui.label("Post-processing");
            ui.add(
                egui::Slider::new(
                    &mut config.post_process_config.chromatic_aberration.intensity,
                    0.0..=0.5,
                )
                .text("Chromatic aberration"),
            );
            ui.add(
                egui::Slider::new(&mut config.post_process_config.bloom.intensity, 0.0..=0.5)
                    .text("Bloom intensity"),
            );

            ui.add(
                egui::Slider::new(&mut config.post_process_config.bloom.scale.x, 0.0..=20.)
                    .text("Bloom X-scale"),
            );
            ui.add(
                egui::Slider::new(&mut config.post_process_config.bloom.scale.y, 0.0..=20.)
                    .text("Bloom Y-scale"),
            );
            ui.add(
                egui::Slider::new(
                    &mut config.post_process_config.depth_of_field.focal_distance,
                    0.0..=80.,
                )
                .text("Focal distance"),
            );
            ui.add(
                egui::Slider::new(
                    &mut config.post_process_config.depth_of_field.max_depth,
                    0.0..=80.,
                )
                .text("Max depth"),
            );

            ui.separator();
            ui.label(format!("Particles: {}", config.num_particles));
            ui.label(format!("Resolution: {}x{}", config.width, config.height));

            ui.separator();
            ui.label("Press F1 to toggle this panel");
        });
}
