use egui::RawInput;
use rust_gpu_bindless::generic::descriptor::{Bindless, BindlessImageUsage, DescriptorCounts};
use rust_gpu_bindless::generic::pipeline::{ClearValue, ColorAttachment, LoadOp, MutImageAccessExt, Present};
use rust_gpu_bindless::generic::platform::ash::Debuggers;
use rust_gpu_bindless::generic::platform::ash::{
	ash_init_single_graphics_queue, Ash, AshSingleGraphicsQueueCreateInfo,
};
use rust_gpu_bindless_egui::renderer::{EguiRenderer, EguiRenderingOptions};
use rust_gpu_bindless_winit::ash::{
	ash_enumerate_required_extensions, AshSwapchain, AshSwapchainParams, SwapchainImageFormatPreference,
};
use rust_gpu_bindless_winit::event_loop::{event_loop_init, EventLoopExecutor};
use rust_gpu_bindless_winit::window_ref::WindowRef;
use std::sync::mpsc::Receiver;
use winit::event::{Event, WindowEvent};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::WindowAttributes;

pub fn main() {
	event_loop_init(|event_loop, events| async {
		main_loop(event_loop, events).await.unwrap();
	});
}

pub fn debugger() -> Debuggers {
	Debuggers::Validation
}

pub async fn main_loop(event_loop: EventLoopExecutor, events: Receiver<Event<()>>) -> anyhow::Result<()> {
	if matches!(debugger(), Debuggers::RenderDoc) {
		// renderdoc does not yet support wayland
		std::env::remove_var("WAYLAND_DISPLAY");
		std::env::set_var("ENABLE_VULKAN_RENDERDOC_CAPTURE", "1");
	}

	let (window, window_extensions) = event_loop
		.spawn(|e| {
			let window = e.create_window(WindowAttributes::default().with_title("swapchain triangle"))?;
			let extensions = ash_enumerate_required_extensions(e.display_handle()?.as_raw())?;
			Ok::<_, anyhow::Error>((WindowRef::new(window), extensions))
		})
		.await?;

	let bindless = unsafe {
		Bindless::<Ash>::new(
			ash_init_single_graphics_queue(AshSingleGraphicsQueueCreateInfo {
				instance_extensions: window_extensions,
				extensions: &[ash::khr::swapchain::NAME, ash::ext::mesh_shader::NAME],
				debug: debugger(),
				..AshSingleGraphicsQueueCreateInfo::default()
			})?,
			DescriptorCounts::REASONABLE_DEFAULTS,
		)
	};

	let mut swapchain = unsafe {
		let bindless2 = bindless.clone();
		AshSwapchain::new(&bindless, &event_loop, &window, move |surface, _| {
			AshSwapchainParams::automatic_best(
				&bindless2,
				surface,
				BindlessImageUsage::COLOR_ATTACHMENT,
				SwapchainImageFormatPreference::SRGB,
			)
		})
	}
	.await?;

	let egui_renderer = EguiRenderer::new(bindless.clone());
	let egui_pipeline = egui_renderer.create_pipeline(Some(swapchain.params().format), None);
	let mut egui_ctx = egui_renderer.create_context(egui::Context::default());

	'outer: loop {
		for event in events.try_iter() {
			swapchain.handle_input(&event);
			if let Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} = &event
			{
				break 'outer;
			}
		}

		let (egui_render, _full_output) = egui_ctx.run(RawInput::default(), ui)?;

		let rt = swapchain.acquire_image(None).await?;
		let rt = bindless.execute(|cmd| {
			let mut rt = rt.access_dont_care::<ColorAttachment>(cmd)?;
			egui_render
				.draw(
					&egui_pipeline,
					cmd,
					&mut rt,
					None,
					EguiRenderingOptions {
						image_rt_load_op: LoadOp::Clear(ClearValue::ColorF([0.; 4])),
						depth_rt_load_op: LoadOp::Load,
					},
				)
				.unwrap();
			Ok(rt.transition::<Present>()?.into_desc())
		})?;
		swapchain.present_image(rt)?;
	}

	Ok(())
}

fn ui(ctx: &egui::Context) {
	egui::SidePanel::left("left panel").show(ctx, |ui| {
		ui.label("Hello world!");
		if ui.button("Button").clicked() {
			println!("Button clicked!");
		}
	});
}
