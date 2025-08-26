use bevy::render::render_resource::ShaderType;

// #[derive(ShaderType)]
pub(crate) enum ColorMode {
	WhiteOnBlack = 0,
	BlueishOrangePurple,
	IcyBlue,
	OrangePurple,
	GoldGreen,
	NeonInferno,
	PinkPurple,
	NeonInfernoArctic,
	BrightYellowBlue,
	Green,
	SmoothBlurryCyan = 10_000
}

#[derive(ShaderType)]
pub(crate) struct UniformData {
	pub(crate) width: u32,
	pub(crate) height: u32,
	pub(crate) value: f32,
	pub(crate) color_mode: u32
}

