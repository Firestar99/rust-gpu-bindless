pub mod ash;
mod bindless;

pub use bindless::*;

/// public interface for a Graphics API. Feel free to use as a base template for other traits.
pub unsafe trait Platform: Sized + Send + Sync + 'static {
	type Entry: 'static;
	type Instance: 'static;
	type PhysicalDevice: 'static;
	type Device: Clone + 'static;
	type MemoryAllocator: 'static;
	type MemoryAllocation: 'static;
	type Buffer: 'static;
	type TypedBuffer<T: Send + Sync + ?Sized + 'static>: 'static;
	type Image: 'static;
	type ImageView: 'static;
	type Sampler: 'static;
	type AllocationError: 'static;
	type DescriptorSet: Clone + 'static;
}
