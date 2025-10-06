use crate::resources::config::PhysarumConfig;
use bevy::post_process::bloom::Bloom;
use bevy::post_process::dof::DepthOfField;
use bevy::post_process::effect_stack::ChromaticAberration;
use bevy::prelude::*;

pub fn update_post_process_settings(
    mut chromatic_aberration: Query<&mut ChromaticAberration>,
    mut bloom: Query<&mut Bloom>,
    mut depth_of_field: Query<&mut DepthOfField>,
    app_settings: Res<PhysarumConfig>,
) {
    let ca_intensity = app_settings
        .post_process_config
        .chromatic_aberration
        .intensity;

    // Pick a reasonable maximum sample size for the intensity to avoid an
    // artifact whereby the individual samples appear instead of producing
    // smooth streaks of color.
    //
    // Don't take this formula too seriously; it hasn't been heavily tuned.
    let max_samples = ((ca_intensity - 0.02) / (0.20 - 0.02) * 56.0 + 8.0)
        .clamp(8.0, 64.0)
        .round() as u32;

    for mut chromatic_aberration in &mut chromatic_aberration {
        chromatic_aberration.intensity = ca_intensity;
        chromatic_aberration.max_samples = max_samples;
    }

    for mut b in &mut bloom {
        *b = app_settings.post_process_config.bloom.clone();
    }

    for mut dof in &mut depth_of_field {
        *dof = app_settings.post_process_config.depth_of_field.clone();
    }
}
