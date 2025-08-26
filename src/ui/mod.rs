use bevy::core_widgets::{Activate, Callback, CoreButton};
use bevy::prelude::*;

use crate::simulation::constants;
use crate::simulation::resources::main::PhysarumInputState;
use crate::simulation::utils::load_parameters;

pub struct UiPlugin;

#[derive(Component)]
struct DebugText;

#[derive(Component)]
struct PresetButton(i32); // -1 for prev, +1 for next

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ui).add_systems(Update, refresh_texts);
    }
}

fn setup_ui(mut commands: Commands) {
    // Register callback systems used by buttons
    let on_preset_click = commands.register_system(on_preset_click);

    // Root UI node
    commands
        .spawn((Node::default(),))
        .with_children(|parent| {
            // Title
            parent.spawn((
                Text::new("Physarum Controls"),
                TextColor(Color::srgb(1.0, 1.0, 1.0)),
            ));

            // Debug line
            parent.spawn((
                Text::new("Debug"),
                TextColor(Color::srgb(1.0, 1.0, 0.0)),
                DebugText,
            ));

            // Preset controls row
            parent
                .spawn((Node::default(),))
                .with_children(|row| {
                    // Prev button
                    row.spawn((
                        Node::default(),
                        CoreButton {
                            on_activate: Callback::System(on_preset_click),
                        },
                        PresetButton(-1),
                        BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                        Children::spawn((Spawn((Text::new("Prev Preset"),)),)),
                    ));

                    // Next button
                    row.spawn((
                        Node::default(),
                        CoreButton {
                            on_activate: Callback::System(on_preset_click),
                        },
                        PresetButton(1),
                        BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                        Children::spawn((Spawn((Text::new("Next Preset"),)),)),
                    ));
                });
        });
}

fn on_preset_click(
    In(Activate(entity)): In<Activate>,
    mut input_state: ResMut<PhysarumInputState>,
    q_preset: Query<&PresetButton>,
) {
    if let Ok(PresetButton(dir)) = q_preset.get(entity) {
        let mut new_index = input_state.new_index as i32 + *dir;
        if new_index < 0 {
            new_index = (constants::NUMBER_OF_BASE_POINTS as i32) - 1;
        }
        new_index %= constants::NUMBER_OF_BASE_POINTS as i32;
        input_state.new_index = new_index as usize;
        input_state.current_settings = load_parameters(input_state.new_index);
        input_state.settings_changed = true;
    }
}

fn refresh_texts(
    mut debug_text_q: Query<&mut Text, With<DebugText>>,
    time: Res<Time>,
    input_state: Res<PhysarumInputState>,
) {
    // Basic FPS estimate from delta time (not averaged)
    if let Ok(mut text) = debug_text_q.single_mut() {
        let fps = if time.delta_secs_f64() > 0.0 {
            (1.0 / time.delta_secs_f64()) as f32
        } else {
            0.0
        };
        *text = Text::new(format!(
            "Preset: {}/{}  |  Particles: {}  |  FPS: {:.0}",
            input_state.new_index + 1,
            constants::NUMBER_OF_BASE_POINTS,
            constants::NUM_PARTICLES,
            fps
        ));
    }
}
