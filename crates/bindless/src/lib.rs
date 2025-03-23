/// The primary platform is Ash
#[cfg(feature = "ash")]
pub type P = rust_gpu_bindless_core::platform::ash::Ash;
#[cfg(not(any(feature = "ash")))]
compile_error!("Must select a primary platform by enabling a feature like \"ash\"");

pub mod backing {
	pub use rust_gpu_bindless_core::backing::primary::*;
}
pub mod descriptor {
	pub use rust_gpu_bindless_core::descriptor::primary::*;
}
pub mod pipeline {
	pub use rust_gpu_bindless_core::pipeline::primary::*;
}
pub mod platform {
	pub use rust_gpu_bindless_core::platform::primary::*;
}
