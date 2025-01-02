use crate::backing::range_set::{DescriptorIndexIterator, DescriptorIndexRangeSet};
use crate::backing::table::DrainFlushQueue;
use crate::descriptor::{
	Bindless, BindlessBufferCreateInfo, BindlessBufferUsage, BindlessImageCreateInfo, BindlessImageUsage,
	BufferAllocationError, BufferInterface, BufferSlot, DescriptorCounts, ImageAllocationError, ImageInterface,
	SamplerInterface,
};
use crate::platform::ash::image_format::FormatExt;
use crate::platform::ash::{
	bindless_image_type_to_vk_image_type, bindless_image_type_to_vk_image_view_type, AshExecutionManager,
};
use crate::platform::BindlessPlatform;
use ash::vk::{
	ComponentMapping, DebugUtilsObjectNameInfoEXT, DescriptorBindingFlags, DescriptorBufferInfo, DescriptorImageInfo,
	DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet,
	DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBindingFlagsCreateInfo,
	DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType, ImageLayout, ImageSubresourceRange,
	ImageTiling, ImageViewCreateInfo, PhysicalDeviceProperties2, PhysicalDeviceVulkan12Properties, PipelineCache,
	PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange, ShaderStageFlags, SharingMode, WriteDescriptorSet,
};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use gpu_allocator::AllocationError;
use parking_lot::lock_api::MutexGuard;
use parking_lot::{Mutex, RawMutex};
use presser::Slab;
use rangemap::RangeSet;
use rust_gpu_bindless_shaders::descriptor::{
	BindlessPushConstant, ImageType, BINDING_BUFFER, BINDING_SAMPLED_IMAGE, BINDING_SAMPLER, BINDING_STORAGE_IMAGE,
};
use static_assertions::assert_impl_all;
use std::cell::UnsafeCell;
use std::ffi::CString;
use std::mem::{size_of, MaybeUninit};
use std::ops::Deref;
use std::sync::Weak;
use thiserror::Error;

pub struct Ash {
	pub create_info: AshCreateInfo,
	pub execution_manager: AshExecutionManager,
}
assert_impl_all!(Bindless<Ash>: Send, Sync);

impl Ash {
	pub fn new(create_info: AshCreateInfo, bindless: &Weak<Bindless<Self>>) -> Self {
		Ash {
			execution_manager: AshExecutionManager::new(bindless),
			create_info,
		}
	}
}

impl Deref for Ash {
	type Target = AshCreateInfo;

	fn deref(&self) -> &Self::Target {
		&self.create_info
	}
}

impl Drop for Ash {
	fn drop(&mut self) {
		self.execution_manager.destroy(&self.create_info.device);
	}
}

pub struct AshCreateInfo {
	pub entry: ash::Entry,
	pub instance: ash::Instance,
	pub physical_device: ash::vk::PhysicalDevice,
	pub device: ash::Device,
	pub memory_allocator: Option<Mutex<Allocator>>,
	pub shader_stages: ShaderStageFlags,
	pub queue_family_index: u32,
	pub queue: ash::vk::Queue,
	pub cache: Option<PipelineCache>,
	pub extensions: AshExtensions,
	pub destroy: Option<Box<dyn FnOnce(&mut AshCreateInfo) + Send + Sync>>,
}

#[derive(Default)]
#[non_exhaustive]
pub struct AshExtensions {
	pub ext_debug_utils: Option<ash::ext::debug_utils::Device>,
	pub ext_mesh_shader: Option<ash::ext::mesh_shader::Device>,
}

impl AshExtensions {
	pub fn ext_mesh_shader(&self) -> &ash::ext::mesh_shader::Device {
		self.ext_mesh_shader.as_ref().expect("missing ext_mesh_shader")
	}
}

impl AshCreateInfo {
	pub fn memory_allocator(&self) -> MutexGuard<'_, RawMutex, Allocator> {
		self.memory_allocator.as_ref().unwrap().lock()
	}
}

impl Drop for AshCreateInfo {
	fn drop(&mut self) {
		if let Some(destroy) = self.destroy.take() {
			destroy(self);
		}
	}
}

/// Wraps gpu-allocator's MemoryAllocation to be able to [`Option::take`] it on drop, but saving the enum flag byte
/// with [`MaybeUninit`]
///
/// # Safety
/// UnsafeCell: Required to gain mutable access where it is safe to do so, see safety of interface methods.
/// MaybeUninit: The Allocation is effectively always initialized, it only becomes uninit after taking it during drop.
#[derive(Debug)]
pub struct AshMemoryAllocation(UnsafeCell<MaybeUninit<Allocation>>);

impl AshMemoryAllocation {
	/// Create a AshMemoryAllocation from a gpu-allocator Allocation
	///
	/// # Safety
	/// You must [`Self::take`] the Allocation and deallocate manually before dropping self
	pub unsafe fn new(allocation: Allocation) -> Self {
		Self(UnsafeCell::new(MaybeUninit::new(allocation)))
	}

	/// Get exclusive mutable access to the Allocation
	///
	/// # Safety
	/// You must ensure you have exclusive mutable access to the Allocation
	pub unsafe fn get_mut(&self) -> &mut Allocation {
		unsafe { (&mut *self.0.get()).assume_init_mut() }
	}

	/// Take the allocation
	///
	/// # Safety
	/// Once the allocation was taken, you must only drop self, any other action is unsafe
	pub unsafe fn take(&self) -> Allocation {
		unsafe { (&mut *self.0.get()).assume_init_read() }
	}
}

/// Safety: MemoryAllocation is safety Send and Sync, will only uninit on drop
unsafe impl Send for AshMemoryAllocation {}
unsafe impl Sync for AshMemoryAllocation {}

pub struct AshBuffer {
	pub buffer: ash::vk::Buffer,
	pub allocation: AshMemoryAllocation,
}

pub struct AshImage {
	pub image: ash::vk::Image,
	pub image_view: Option<ash::vk::ImageView>,
	pub allocation: AshMemoryAllocation,
}

#[derive(Copy, Clone, Debug)]
pub struct AshBindlessDescriptorSet {
	pub pipeline_layout: PipelineLayout,
	pub set_layout: DescriptorSetLayout,
	pub pool: DescriptorPool,
	pub set: DescriptorSet,
}

impl Deref for AshBindlessDescriptorSet {
	type Target = DescriptorSet;

	fn deref(&self) -> &Self::Target {
		&self.set
	}
}

#[derive(Error)]
pub enum AshAllocationError {
	#[error("VkResult: {0}")]
	Vk(#[from] ash::vk::Result),
	#[error("gpu-allocator Error: {0}")]
	Allocation(#[from] AllocationError),
}

impl core::fmt::Debug for AshAllocationError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Display::fmt(self, f)
	}
}

impl From<AshAllocationError> for BufferAllocationError<Ash> {
	fn from(value: AshAllocationError) -> Self {
		BufferAllocationError::Platform(value)
	}
}

impl From<AshAllocationError> for ImageAllocationError<Ash> {
	fn from(value: AshAllocationError) -> Self {
		ImageAllocationError::Platform(value)
	}
}

unsafe impl BindlessPlatform for Ash {
	type PlatformCreateInfo = AshCreateInfo;
	type Buffer = AshBuffer;
	type Image = AshImage;
	type Sampler = ash::vk::Sampler;
	type AllocationError = AshAllocationError;
	type BindlessDescriptorSet = AshBindlessDescriptorSet;

	unsafe fn create_platform(create_info: Self::PlatformCreateInfo, bindless_cyclic: &Weak<Bindless<Self>>) -> Self {
		Ash::new(create_info, bindless_cyclic)
	}

	unsafe fn update_after_bind_descriptor_limits(&self) -> DescriptorCounts {
		let mut vulkan12properties = PhysicalDeviceVulkan12Properties::default();
		let mut properties2 = PhysicalDeviceProperties2::default().push_next(&mut vulkan12properties);
		self.instance
			.get_physical_device_properties2(self.physical_device, &mut properties2);
		DescriptorCounts {
			buffers: vulkan12properties.max_descriptor_set_update_after_bind_storage_buffers,
			image: u32::min(
				vulkan12properties.max_per_stage_descriptor_update_after_bind_storage_images,
				vulkan12properties.max_descriptor_set_update_after_bind_sampled_images,
			),
			samplers: vulkan12properties.max_descriptor_set_update_after_bind_samplers,
		}
	}

	unsafe fn create_descriptor_set(&self, counts: DescriptorCounts) -> Self::BindlessDescriptorSet {
		let bindings = [
			ash::vk::DescriptorSetLayoutBinding::default()
				.binding(BINDING_BUFFER)
				.descriptor_type(DescriptorType::STORAGE_BUFFER)
				.descriptor_count(counts.buffers)
				.stage_flags(self.shader_stages),
			ash::vk::DescriptorSetLayoutBinding::default()
				.binding(BINDING_STORAGE_IMAGE)
				.descriptor_type(DescriptorType::STORAGE_IMAGE)
				.descriptor_count(counts.image)
				.stage_flags(self.shader_stages),
			ash::vk::DescriptorSetLayoutBinding::default()
				.binding(BINDING_SAMPLED_IMAGE)
				.descriptor_type(DescriptorType::SAMPLED_IMAGE)
				.descriptor_count(counts.image)
				.stage_flags(self.shader_stages),
			ash::vk::DescriptorSetLayoutBinding::default()
				.binding(BINDING_SAMPLER)
				.descriptor_type(DescriptorType::SAMPLER)
				.descriptor_count(counts.samplers)
				.stage_flags(self.shader_stages),
		];
		let binding_flags =
			[DescriptorBindingFlags::UPDATE_AFTER_BIND | DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING; 4];
		assert_eq!(bindings.len(), binding_flags.len());

		let set_layout = self
			.device
			.create_descriptor_set_layout(
				&DescriptorSetLayoutCreateInfo::default()
					.flags(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
					.bindings(&bindings)
					.push_next(&mut DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags)),
				None,
			)
			.unwrap();

		let pipeline_layout = self
			.device
			.create_pipeline_layout(
				&PipelineLayoutCreateInfo::default()
					.set_layouts(&[set_layout])
					.push_constant_ranges(&[PushConstantRange {
						offset: 0,
						size: size_of::<BindlessPushConstant>() as u32,
						stage_flags: self.shader_stages,
					}]),
				None,
			)
			.unwrap();

		let pool = self
			.device
			.create_descriptor_pool(
				&DescriptorPoolCreateInfo::default()
					.flags(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
					.pool_sizes(&bindings.map(|b| {
						DescriptorPoolSize::default()
							.ty(b.descriptor_type)
							.descriptor_count(b.descriptor_count)
					}))
					.max_sets(1),
				None,
			)
			.unwrap();

		let set = self
			.device
			.allocate_descriptor_sets(
				&DescriptorSetAllocateInfo::default()
					.descriptor_pool(pool)
					.set_layouts(&[set_layout]),
			)
			.unwrap()
			.into_iter()
			.next()
			.unwrap();

		AshBindlessDescriptorSet {
			pipeline_layout,
			set_layout,
			pool,
			set,
		}
	}

	unsafe fn bindless_initialized(&self, _bindless: &mut Bindless<Self>) {}

	unsafe fn update_descriptor_set(
		&self,
		set: &Self::BindlessDescriptorSet,
		mut buffers: DrainFlushQueue<BufferInterface<Self>>,
		mut images: DrainFlushQueue<ImageInterface<Self>>,
		mut samplers: DrainFlushQueue<SamplerInterface<Self>>,
	) {
		let (buffer_table, buffers) = buffers.into_inner();
		let mut storage_buffers = DescriptorIndexRangeSet::new(buffer_table, RangeSet::new());
		for buffer_id in buffers {
			let buffer = unsafe { buffer_table.get_slot_unchecked(buffer_id) };
			if buffer.usage.contains(BindlessBufferUsage::STORAGE_BUFFER) {
				storage_buffers.insert(buffer_id);
			}
		}

		let buffer_infos = storage_buffers
			.iter()
			.map(|(_, buffer)| {
				DescriptorBufferInfo::default()
					.buffer(buffer.buffer)
					.offset(0)
					.range(buffer.size)
			})
			.collect::<Vec<_>>();
		let mut buffer_info_index = 0;
		let buffers = storage_buffers.iter_ranges().map(|(range, _)| {
			let count = range.end.to_usize() - range.start.to_usize();
			WriteDescriptorSet::default()
				.dst_set(set.set)
				.dst_binding(BINDING_BUFFER)
				.descriptor_type(DescriptorType::STORAGE_BUFFER)
				.dst_array_element(range.start.to_u32())
				.descriptor_count(count as u32)
				.buffer_info({
					let buffer_info_start = buffer_info_index;
					buffer_info_index += count;
					&buffer_infos[buffer_info_start..buffer_info_start + count]
				})
		});

		let (image_table, images) = images.into_inner();
		let mut storage_images = DescriptorIndexRangeSet::new(image_table, RangeSet::new());
		let mut sampled_images = DescriptorIndexRangeSet::new(image_table, RangeSet::new());
		for image_id in images {
			let image = unsafe { image_table.get_slot_unchecked(image_id) };
			if image.usage.contains(BindlessImageUsage::STORAGE) {
				storage_images.insert(image_id);
			}
			if image.usage.contains(BindlessImageUsage::SAMPLED) {
				sampled_images.insert(image_id);
			}
		}

		let storage_image_infos = storage_images
			.iter()
			.map(|(_, storage_image)| {
				DescriptorImageInfo::default()
					.image_view(*storage_image.image_view.as_ref().unwrap())
					.image_layout(ImageLayout::GENERAL)
			})
			.collect::<Vec<_>>();
		let mut storage_image_info_index = 0;
		let storage_images = storage_images.iter_ranges().map(|(range, _)| {
			let count = range.end.to_usize() - range.start.to_usize();
			WriteDescriptorSet::default()
				.dst_set(set.set)
				.dst_binding(BINDING_STORAGE_IMAGE)
				.descriptor_type(DescriptorType::STORAGE_IMAGE)
				.dst_array_element(range.start.to_u32())
				.descriptor_count(count as u32)
				.image_info({
					let storage_image_info_start = storage_image_info_index;
					storage_image_info_index += count;
					&storage_image_infos[storage_image_info_start..storage_image_info_start + count]
				})
		});

		let sampled_image_infos = sampled_images
			.iter()
			.map(|(_, sampled_image)| {
				DescriptorImageInfo::default()
					.image_view(*sampled_image.image_view.as_ref().unwrap())
					.image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
			})
			.collect::<Vec<_>>();
		let mut sampled_image_info_index = 0;
		let sampled_images = sampled_images.iter_ranges().map(|(range, _)| {
			let count = range.end.to_usize() - range.start.to_usize();
			WriteDescriptorSet::default()
				.dst_set(set.set)
				.dst_binding(BINDING_SAMPLED_IMAGE)
				.descriptor_type(DescriptorType::SAMPLED_IMAGE)
				.dst_array_element(range.start.to_u32())
				.descriptor_count(count as u32)
				.image_info({
					let sampled_image_info_start = sampled_image_info_index;
					sampled_image_info_index += count;
					&sampled_image_infos[sampled_image_info_start..sampled_image_info_start + count]
				})
		});

		let samplers = samplers.into_range_set();
		let sampler_infos = samplers
			.iter()
			.map(|(_, sampler)| DescriptorImageInfo::default().sampler(*sampler))
			.collect::<Vec<_>>();
		let mut sampler_info_index = 0;
		let samplers = samplers.iter_ranges().map(|(range, _)| {
			let count = range.end.to_usize() - range.start.to_usize();
			WriteDescriptorSet::default()
				.dst_set(set.set)
				.dst_binding(BINDING_SAMPLER)
				.descriptor_type(DescriptorType::SAMPLER)
				.dst_array_element(range.start.to_u32())
				.descriptor_count(count as u32)
				.image_info({
					let sampler_info_start = sampler_info_index;
					sampler_info_index += count;
					&sampler_infos[sampler_info_start..sampler_info_start + count]
				})
		});

		let writes = buffers
			.chain(storage_images)
			.chain(sampled_images)
			.chain(samplers)
			.collect::<Vec<_>>();
		self.device.update_descriptor_sets(&writes, &[]);
	}

	unsafe fn destroy_descriptor_set(&self, set: Self::BindlessDescriptorSet) {
		// descriptor sets allocated from pool are freed implicitly
		self.device.destroy_descriptor_pool(set.pool, None);
		self.device.destroy_pipeline_layout(set.pipeline_layout, None);
		self.device.destroy_descriptor_set_layout(set.set_layout, None);
	}

	unsafe fn alloc_buffer(
		&self,
		create_info: &BindlessBufferCreateInfo,
		size: u64,
	) -> Result<Self::Buffer, Self::AllocationError> {
		let buffer = self.device.create_buffer(
			&ash::vk::BufferCreateInfo::default()
				.usage(create_info.usage.to_ash_buffer_usage_flags())
				.size(size)
				.sharing_mode(SharingMode::EXCLUSIVE),
			None,
		)?;
		if let Some(debug_marker) = self.extensions.ext_debug_utils.as_ref() {
			debug_marker.set_debug_utils_object_name(
				&DebugUtilsObjectNameInfoEXT::default()
					.object_handle(buffer)
					.object_name(&CString::new(create_info.name).unwrap()),
			)?;
		}
		let requirements = self.device.get_buffer_memory_requirements(buffer);
		let memory_allocation = self.memory_allocator().allocate(&AllocationCreateDesc {
			requirements,
			name: create_info.name,
			location: create_info.usage.to_gpu_allocator_memory_location(),
			allocation_scheme: create_info.allocation_scheme.to_gpu_allocator_buffer(buffer),
			linear: true,
		})?;
		self.device
			.bind_buffer_memory(buffer, memory_allocation.memory(), memory_allocation.offset())?;
		Ok(AshBuffer {
			buffer,
			allocation: AshMemoryAllocation::new(memory_allocation),
		})
	}

	unsafe fn alloc_image<T: ImageType>(
		&self,
		create_info: &BindlessImageCreateInfo<T>,
	) -> Result<Self::Image, Self::AllocationError> {
		let image_type = bindless_image_type_to_vk_image_type::<T>().expect("Unsupported ImageType");
		let image_view_type = bindless_image_type_to_vk_image_view_type::<T>().expect("Unsupported ImageType");
		let image = self.device.create_image(
			&ash::vk::ImageCreateInfo::default()
				.flags(ash::vk::ImageCreateFlags::empty())
				.image_type(image_type)
				.format(create_info.format)
				.extent(create_info.extent.into())
				.mip_levels(create_info.mip_levels)
				.array_layers(create_info.array_layers)
				.samples(create_info.samples.to_ash_sample_count_flags())
				.tiling(ImageTiling::OPTIMAL)
				.usage(create_info.usage.to_ash_image_usage_flags())
				.sharing_mode(SharingMode::EXCLUSIVE)
				.initial_layout(ImageLayout::UNDEFINED),
			None,
		)?;
		if let Some(debug_marker) = self.extensions.ext_debug_utils.as_ref() {
			debug_marker.set_debug_utils_object_name(
				&DebugUtilsObjectNameInfoEXT::default()
					.object_handle(image)
					.object_name(&CString::new(create_info.name).unwrap()),
			)?;
		}
		let requirements = self.device.get_image_memory_requirements(image);
		let memory_allocation = self.memory_allocator().allocate(&AllocationCreateDesc {
			requirements,
			name: create_info.name,
			location: create_info.usage.to_gpu_allocator_memory_location(),
			allocation_scheme: create_info.allocation_scheme.to_gpu_allocator_image(image),
			linear: true,
		})?;
		self.device
			.bind_image_memory(image, memory_allocation.memory(), memory_allocation.offset())?;
		let image_view = if create_info.usage.has_image_view() {
			let image_view = self.device.create_image_view(
				&ImageViewCreateInfo::default()
					.image(image)
					.view_type(image_view_type)
					.format(create_info.format)
					.components(ComponentMapping::default()) // identity
					.subresource_range(ImageSubresourceRange {
						aspect_mask: create_info.format.aspect(),
						base_mip_level: 0,
						level_count: create_info.mip_levels,
						base_array_layer: 0,
						layer_count: create_info.array_layers,
					}),
				None,
			)?;
			if let Some(debug_marker) = self.extensions.ext_debug_utils.as_ref() {
				debug_marker.set_debug_utils_object_name(
					&DebugUtilsObjectNameInfoEXT::default()
						.object_handle(image_view)
						.object_name(&CString::new(create_info.name).unwrap()),
				)?;
			}
			Some(image_view)
		} else {
			None
		};
		Ok(AshImage {
			image,
			image_view,
			allocation: AshMemoryAllocation::new(memory_allocation),
		})
	}

	unsafe fn mapped_buffer_to_slab<'a>(buffer: &'a BufferSlot<Self>) -> &'a mut (impl Slab + 'a) {
		buffer.allocation.get_mut()
	}

	unsafe fn destroy_buffers<'a>(
		&self,
		_global_descriptor_set: &Self::BindlessDescriptorSet,
		buffers: impl DescriptorIndexIterator<'a, BufferInterface<Self>>,
	) {
		let mut allocator = self.memory_allocator();
		for (_, buffer) in buffers.into_iter() {
			// Safety: We have exclusive access to BufferSlot in this method. The MemoryAllocation will no longer
			// we accessed by anything nor dropped due to being wrapped in MaybeUninit, so we can safely read and drop
			// it ourselves.
			let allocation = buffer.allocation.take();
			allocator.free(allocation).unwrap();
			self.device.destroy_buffer(buffer.buffer, None);
		}
	}

	unsafe fn destroy_images<'a>(
		&self,
		_global_descriptor_set: &Self::BindlessDescriptorSet,
		images: impl DescriptorIndexIterator<'a, ImageInterface<Self>>,
	) {
		let mut allocator = self.memory_allocator();
		for (_, image) in images.into_iter() {
			// Safety: We have exclusive access to BufferSlot in this method. The MemoryAllocation will no longer
			// we accessed by anything nor dropped due to being wrapped in MaybeUninit, so we can safely read and drop
			// it ourselves.
			let allocation = image.allocation.take();
			allocator.free(allocation).unwrap();
			if let Some(imageview) = image.image_view {
				self.device.destroy_image_view(imageview, None);
			}
			self.device.destroy_image(image.image, None);
		}
	}

	unsafe fn destroy_samplers<'a>(
		&self,
		_global_descriptor_set: &Self::BindlessDescriptorSet,
		samplers: impl DescriptorIndexIterator<'a, SamplerInterface<Self>>,
	) {
		for (_, sampler) in samplers.into_iter() {
			self.device.destroy_sampler(*sampler, None);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// we want our [`BindlessBufferUsage`] bits to equal ash's [`BufferUsageFlags`] so the conversion can mostly be
	/// optimized away
	#[test]
	fn test_buffer_usage_to_ash_same_bits() {
		for usage in [
			BindlessBufferUsage::TRANSFER_SRC,
			BindlessBufferUsage::TRANSFER_DST,
			BindlessBufferUsage::UNIFORM_BUFFER,
			BindlessBufferUsage::STORAGE_BUFFER,
			BindlessBufferUsage::INDEX_BUFFER,
			BindlessBufferUsage::VERTEX_BUFFER,
			BindlessBufferUsage::INDIRECT_BUFFER,
		] {
			assert_eq!(
				Some(usage),
				BindlessBufferUsage::from_bits(usage.to_ash_buffer_usage_flags().as_raw() as u64)
			)
		}
	}
}
