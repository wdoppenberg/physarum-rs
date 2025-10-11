use bevy::asset::{Assets, RenderAssetUsages};
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::WindowResized;

use crate::resources::config::PhysarumConfig;
use crate::resources::render::PhysarumImages;

/// System to handle window resize events and update the simulation accordingly
pub fn handle_window_resize(
    mut resize_events: MessageReader<WindowResized>,
    mut config: ResMut<PhysarumConfig>,
    mut images: ResMut<Assets<Image>>,
    mut physarum_images: Option<ResMut<PhysarumImages>>,
    mut sprite_query: Query<&mut Sprite>,
) {
    // Process only the last resize event to avoid redundant work
    let last_event = resize_events.read().last();
    
    if let Some(event) = last_event {
        let new_width = event.width as u32;
        let new_height = event.height as u32;

        // Skip if dimensions haven't actually changed
        if new_width == config.width && new_height == config.height {
            return;
        }

        info!(
            "Window resized to {}x{}, updating simulation",
            new_width, new_height
        );

        // Update config dimensions
        config.width = new_width;
        config.height = new_height;

        // Recreate textures if PhysarumImages resource exists
        if let Some(ref mut physarum_images) = physarum_images {
            // Create new display texture
            let mut display_image = Image::new_fill(
                Extent3d {
                    width: new_width,
                    height: new_height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                &[0, 0, 0, 255],
                TextureFormat::Rgba8Unorm,
                RenderAssetUsages::RENDER_WORLD,
            );
            display_image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST;

            // Create new trail textures
            let mut trail_texture = Image::new_fill(
                Extent3d {
                    width: new_width,
                    height: new_height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                &vec![0; (new_width * new_height * 4) as usize],
                TextureFormat::R32Float,
                RenderAssetUsages::RENDER_WORLD,
            );
            trail_texture.texture_descriptor.usage |=
                TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

            // Replace existing textures with new ones
            let new_display_texture = images.add(display_image);
            let new_texture_a = images.add(trail_texture.clone());
            let new_texture_b = images.add(trail_texture);

            // Remove old textures
            images.remove(&physarum_images.display_texture);
            images.remove(&physarum_images.texture_a);
            images.remove(&physarum_images.texture_b);

            // Update the resource with new texture handles
            physarum_images.display_texture = new_display_texture.clone();
            physarum_images.texture_a = new_texture_a;
            physarum_images.texture_b = new_texture_b;

            // Update sprite size and texture
            for mut sprite in sprite_query.iter_mut() {
                sprite.image = new_display_texture.clone();
                sprite.custom_size = Some(Vec2::new(new_width as f32, new_height as f32));
            }
        }
    }
}
