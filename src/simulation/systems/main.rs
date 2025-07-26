use bevy::prelude::{KeyCode, Res, ResMut};
use bevy::input::ButtonInput;
use bevy::log::info;
use crate::simulation::resources::main::PhysarumInputState;

/// Handle keyboard input to change simulation parameters (main world)
pub fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut input_state: ResMut<PhysarumInputState>,
) {
    let mut changed = false;
    let mut new_index = input_state.new_index;

    if keys.just_pressed(KeyCode::ArrowRight) || keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::Space) {
        new_index = (new_index + 1) % crate::points_basematrix::NUMBER_OF_BASE_POINTS;
        changed = true;
    }

    if keys.just_pressed(KeyCode::ArrowLeft) || keys.just_pressed(KeyCode::ArrowDown) {
        new_index = (new_index + crate::points_basematrix::NUMBER_OF_BASE_POINTS - 1)
            % crate::points_basematrix::NUMBER_OF_BASE_POINTS;
        changed = true;
    }

    if changed {
        info!("Simulation settings changed to {}", new_index);
        input_state.settings_changed = true;
        input_state.new_index = new_index;
    }
}
