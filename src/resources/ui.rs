use bevy::prelude::*;

/// Resource to track UI state
#[derive(Resource, Default)]
pub struct UiState {
    /// Whether the sidebar is visible
    pub sidebar_visible: bool,
}
