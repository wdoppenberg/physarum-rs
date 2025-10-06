use crate::constants::PARAMETERS_MATRIX;
use crate::resources::config::PhysarumConfig;
use crate::resources::input::PhysarumInputState;
use crate::utils::load_parameters;
use bevy::input::ButtonInput;
use bevy::log::info;
use bevy::prelude::{KeyCode, Query, Res, ResMut, Time, Vec2, With};
use bevy::window::{PrimaryWindow, Window};

/// Handle keyboard/mouse input to change simulation parameters and interactive uniforms (main world)
pub fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut input_state: ResMut<PhysarumInputState>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
    mut config: ResMut<PhysarumConfig>,
) {
    // 1) Accumulate time
    input_state.time += time.delta_secs();

    // 2) Mouse position to action coordinates
    if let Ok(win) = q_windows.single() {
        if let Some(pos) = win.cursor_position() {
            // Map window cursor coordinates proportionally into simulation texture coordinates
            // Use current config width/height to scale into simulation texture space.
            let sim_width = config.width as f32;
            let sim_height = config.height as f32;

            // Cursor position is in window space [0..win.width/height]. Scale to [0..sim_width/height].
            // Flip Y to match texture coordinate space (y increasing downward in the simulation textures).
            let x = (pos.x / win.width() * sim_width).clamp(0.0, sim_width - 1.0);
            let y = (pos.y / win.height() * sim_height).clamp(0.0, sim_height - 1.0);

            input_state.action_x = x;
            input_state.action_y = y;
        }
    }

    // 3) Keyboard controls
    let mut changed_params_index = false;
    let mut new_index = config.new_index;

    // Cycle through presets
    if keys.just_pressed(KeyCode::ArrowRight)
        || keys.just_pressed(KeyCode::ArrowUp)
        || keys.just_pressed(KeyCode::Space)
        || keys.just_pressed(KeyCode::KeyR)
    {
        new_index = (new_index + 1) % PARAMETERS_MATRIX.len();
        changed_params_index = true;
    }
    if keys.just_pressed(KeyCode::ArrowLeft) || keys.just_pressed(KeyCode::ArrowDown) {
        new_index = (new_index + PARAMETERS_MATRIX.len() - 1) % PARAMETERS_MATRIX.len();
        changed_params_index = true;
    }

    if changed_params_index {
        info!("Simulation settings changed to {}", new_index);
        config.settings_changed = true;
        config.new_index = new_index;
        config.current_settings = load_parameters(new_index);
    }

    // Movement bias with WASD
    let mut bias = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        bias.y += 10.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        bias.y -= 10.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        bias.x -= 10.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        bias.x += 10.0;
    }
    if bias.length_squared() > 0.0 {
        bias = bias.normalize();
    }
    input_state.move_bias_action_x = bias.x;
    input_state.move_bias_action_y = bias.y;
    input_state.l2_action = bias.length();

    // Spawn triggers
    if keys.just_pressed(KeyCode::KeyF) {
        input_state.spawn_particles = 1; // circular spawn around action
    }
    // Let the spawn flag last only one frame (handled here by clearing if no key press this frame)
    if !(keys.just_pressed(KeyCode::KeyF) || keys.just_pressed(KeyCode::KeyD)) {
        input_state.spawn_particles = 0;
    }

    // Cycle color modes with P
    if keys.just_pressed(KeyCode::KeyP) {
        config.color_mode = (config.color_mode + 1) % 10; // cycle through first 10 modes
    }
}
