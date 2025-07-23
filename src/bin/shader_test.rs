use std::fs;
use wgpu::{Device, Instance, RequestAdapterOptions, DeviceDescriptor, Features, Limits, ShaderModule, ShaderModuleDescriptor, ShaderSource};

async fn compile_shader(device: &Device, shader_path: &str) -> Result<ShaderModule, String> {
    let shader_source = fs::read_to_string(shader_path)
        .map_err(|e| format!("Failed to read shader file {}: {}", shader_path, e))?;
    
    let shader_desc = ShaderModuleDescriptor {
        label: Some(shader_path),
        source: ShaderSource::Wgsl(shader_source.into()),
    };
    
    let shader_module = device.create_shader_module(shader_desc);

    Ok(shader_module)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize wgpu
    let instance = Instance::default();
    
    // Get the default adapter
    let adapter = instance.request_adapter(&RequestAdapterOptions::default())
        .await
        .ok_or("Failed to find an appropriate adapter")?;
    
    println!("Using adapter: {:?}", adapter.get_info());
    
    // Create the device and queue
    let (device, _queue) = adapter.request_device(
        &DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            required_limits: Limits::default(),
        },
        None,
    ).await?;
    
    // List of shaders to test
    let shader_paths = [
        "shaders/setter.wgsl",
        "shaders/move.wgsl",
        "shaders/deposit.wgsl",
        "shaders/diffusion.wgsl",
    ];
    
    // Test each shader
    for shader_path in &shader_paths {
        println!("Testing shader: {}", shader_path);
        
        match compile_shader(&device, shader_path).await {
            Ok(_) => println!("✅ Shader compiled successfully"),
            Err(e) => println!("❌ Shader compilation failed: {}", e),
        }
        
        println!();
    }
    
    Ok(())
}