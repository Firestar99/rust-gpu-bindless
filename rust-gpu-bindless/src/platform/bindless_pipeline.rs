use crate::descriptor::{Bindless, BufferSlot, ImageSlot};
use crate::pipeline::access_buffer::MutBufferAccess;
use crate::pipeline::access_image::MutImageAccess;
use crate::pipeline::access_type::{
	BufferAccess, BufferAccessType, ColorAttachment, DepthStencilAttachment, ImageAccess, ImageAccessType,
	IndexReadable, IndirectCommandReadable, TransferReadable, TransferWriteable,
};
use crate::pipeline::compute_pipeline::BindlessComputePipeline;
use crate::pipeline::graphics_pipeline::{BindlessGraphicsPipeline, GraphicsPipelineCreateInfo};
use crate::pipeline::mesh_graphics_pipeline::{BindlessMeshGraphicsPipeline, MeshGraphicsPipelineCreateInfo};
use crate::pipeline::mut_or_shared::MutOrSharedBuffer;
use crate::pipeline::recording::{HasResourceContext, Recording, RecordingError};
use crate::pipeline::rendering::{IndexTypeTrait, RenderPassFormat, RenderingAttachment};
use crate::pipeline::shader::BindlessShader;
use crate::platform::BindlessPlatform;
use crate::spirv_std::indirect_command::{DrawIndexedIndirectCommand, DrawIndirectCommand};
use rust_gpu_bindless_shaders::buffer_content::{BufferContent, BufferStruct};
use rust_gpu_bindless_shaders::descriptor::{ImageType, TransientAccess};
use rust_gpu_bindless_shaders::shader_type::{ComputeShader, FragmentShader, MeshShader, TaskShader, VertexShader};
use std::error::Error;
use std::sync::Arc;

/// Internal interface for pipeline module related API calls, may change at any time!
pub unsafe trait BindlessPipelinePlatform: BindlessPlatform {
	type PipelineCreationError: 'static + Error + Send + Sync;
	type ComputePipeline: 'static + Send + Sync;
	type RecordingResourceContext: RecordingResourceContext<Self>;
	type RecordingContext<'a>: RecordingContext<'a, Self>;
	type RecordingError: 'static + Error + Send + Sync + Into<RecordingError<Self>>;
	type ExecutingContext<R: Send + Sync>: ExecutingContext<Self, R>;

	unsafe fn create_compute_pipeline<T: BufferStruct>(
		bindless: &Arc<Bindless<Self>>,
		compute_shader: &impl BindlessShader<ShaderType = ComputeShader, ParamConstant = T>,
	) -> Result<Self::ComputePipeline, Self::PipelineCreationError>;

	unsafe fn record_and_execute<R: Send + Sync>(
		bindless: &Arc<Bindless<Self>>,
		f: impl FnOnce(&mut Recording<'_, Self>) -> Result<R, RecordingError<Self>>,
	) -> Result<Self::ExecutingContext<R>, RecordingError<Self>>;

	type GraphicsPipeline: 'static + Send + Sync;
	type MeshGraphicsPipeline: 'static + Send + Sync;
	type RenderingContext<'a: 'b, 'b>: RenderingContext<'a, 'b, Self>;

	unsafe fn create_graphics_pipeline<T: BufferStruct>(
		bindless: &Arc<Bindless<Self>>,
		render_pass: &RenderPassFormat,
		create_info: &GraphicsPipelineCreateInfo,
		vertex_shader: &impl BindlessShader<ShaderType = VertexShader, ParamConstant = T>,
		fragment_shader: &impl BindlessShader<ShaderType = FragmentShader, ParamConstant = T>,
	) -> Result<Self::GraphicsPipeline, Self::PipelineCreationError>;

	unsafe fn create_mesh_graphics_pipeline<T: BufferStruct>(
		bindless: &Arc<Bindless<Self>>,
		render_pass: &RenderPassFormat,
		create_info: &MeshGraphicsPipelineCreateInfo,
		task_shader: Option<&impl BindlessShader<ShaderType = TaskShader, ParamConstant = T>>,
		mesh_shader: &impl BindlessShader<ShaderType = MeshShader, ParamConstant = T>,
		fragment_shader: &impl BindlessShader<ShaderType = FragmentShader, ParamConstant = T>,
	) -> Result<Self::MeshGraphicsPipeline, Self::PipelineCreationError>;
}

pub unsafe trait RecordingContext<'a, P: BindlessPipelinePlatform>: HasResourceContext<'a, P> {
	/// Copy data from a buffer to an image. It is assumed that the image data is tightly packed within the buffer.
	/// Partial copies and copying to mips other than mip 0 is not yet possible.
	unsafe fn copy_buffer_to_image<
		BT: BufferContent + ?Sized,
		BA: BufferAccessType + TransferReadable,
		IT: ImageType,
		IA: ImageAccessType + TransferWriteable,
	>(
		&mut self,
		src: &mut MutBufferAccess<P, BT, BA>,
		dst: &mut MutImageAccess<P, IT, IA>,
	) -> Result<(), P::RecordingError>;

	/// Copy data from an image to a buffer. It is assumed that the image data is tightly packed within the buffer.
	/// Partial copies and copying to mips other than mip 0 is not yet possible.
	///
	/// # Safety
	/// This allows any data to be written to the buffer, without checking the buffer's type, potentially transmuting
	/// data.
	unsafe fn copy_image_to_buffer<
		IT: ImageType,
		IA: ImageAccessType + TransferReadable,
		BT: BufferContent + ?Sized,
		BA: BufferAccessType + TransferWriteable,
	>(
		&mut self,
		src: &mut MutImageAccess<P, IT, IA>,
		dst: &mut MutBufferAccess<P, BT, BA>,
	) -> Result<(), P::RecordingError>;

	/// Dispatch a bindless compute shader
	unsafe fn dispatch<T: BufferStruct>(
		&mut self,
		pipeline: &BindlessComputePipeline<P, T>,
		group_counts: [u32; 3],
		param: T,
	) -> Result<(), P::RecordingError>;

	/// Dispatch a bindless compute shader
	unsafe fn dispatch_indirect<T: BufferStruct, A: BufferAccessType + IndirectCommandReadable>(
		&mut self,
		pipeline: &BindlessComputePipeline<P, T>,
		indirect: impl MutOrSharedBuffer<P, [u32; 3], A>,
		param: T,
	) -> Result<(), P::RecordingError>;
}

pub unsafe trait RecordingResourceContext<P: BindlessPipelinePlatform>: 'static {
	unsafe fn to_transient_access(&self) -> impl TransientAccess<'_>;
	unsafe fn transition_buffer(&self, buffer: &BufferSlot<P>, src: BufferAccess, dst: BufferAccess);

	unsafe fn transition_image(&self, image: &ImageSlot<P>, src: ImageAccess, dst: ImageAccess);
}

pub unsafe trait RenderingContext<'a, 'b, P: BindlessPipelinePlatform>: HasResourceContext<'a, P> {
	unsafe fn begin_rendering(
		recording: &'b mut P::RecordingContext<'a>,
		format: RenderPassFormat,
		render_area: [u32; 2],
		color_attachments: &[RenderingAttachment<P, ColorAttachment>],
		depth_attachment: Option<RenderingAttachment<P, DepthStencilAttachment>>,
	) -> Result<Self, P::RecordingError>;

	unsafe fn end_rendering(&mut self) -> Result<(), P::RecordingError>;

	unsafe fn draw<T: BufferStruct>(
		&mut self,
		pipeline: &BindlessGraphicsPipeline<P, T>,
		count: DrawIndirectCommand,
		param: T,
	) -> Result<(), P::RecordingError>;

	unsafe fn draw_indexed<T: BufferStruct, IT: IndexTypeTrait, AIR: IndexReadable>(
		&mut self,
		pipeline: &BindlessGraphicsPipeline<P, T>,
		index_buffer: impl MutOrSharedBuffer<P, [IT], AIR>,
		count: DrawIndexedIndirectCommand,
		param: T,
	) -> Result<(), RecordingError<P>>;

	unsafe fn draw_indirect<T: BufferStruct, AIC: IndirectCommandReadable>(
		&mut self,
		pipeline: &BindlessGraphicsPipeline<P, T>,
		indirect: impl MutOrSharedBuffer<P, [DrawIndirectCommand], AIC>,
		param: T,
	) -> Result<(), P::RecordingError>;

	unsafe fn draw_indexed_indirect<
		T: BufferStruct,
		IT: IndexTypeTrait,
		AIR: IndexReadable,
		AIC: IndirectCommandReadable,
	>(
		&mut self,
		pipeline: &BindlessGraphicsPipeline<P, T>,
		index_buffer: impl MutOrSharedBuffer<P, [IT], AIR>,
		indirect: impl MutOrSharedBuffer<P, [DrawIndirectCommand], AIC>,
		param: T,
	) -> Result<(), P::RecordingError>;

	unsafe fn draw_mesh_tasks<T: BufferStruct>(
		&mut self,
		pipeline: &BindlessMeshGraphicsPipeline<P, T>,
		group_counts: [u32; 3],
		param: T,
	) -> Result<(), P::RecordingError>;

	unsafe fn draw_mesh_tasks_indirect<T: BufferStruct, AIC: IndirectCommandReadable>(
		&mut self,
		pipeline: &BindlessMeshGraphicsPipeline<P, T>,
		indirect: impl MutOrSharedBuffer<P, [[u32; 3]], AIC>,
		param: T,
	) -> Result<(), P::RecordingError>;
}

pub unsafe trait ExecutingContext<P: BindlessPipelinePlatform, R: Send + Sync>: Send + Sync {
	/// Stopgap solution to wait for execution to finish
	fn block_on(self) -> R;
}
