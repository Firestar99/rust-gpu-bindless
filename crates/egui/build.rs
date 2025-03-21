use rust_gpu_bindless_shader_builder::ShaderSymbolsBuilder;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
	ShaderSymbolsBuilder::new("egui-shaders", "spirv-unknown-vulkan1.2").build()?;
	Ok(())
}
