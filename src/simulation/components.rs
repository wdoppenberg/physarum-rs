use bevy::prelude::*;

/// Component for displaying the physarum simulation
#[derive(Component)]
pub struct PhysarumDisplay {
    /// Handle to the image that will be updated from the compute shader
    pub image_handle: Handle<Image>,
}