use crate::buffer_content::{BufferContent, BufferStruct, Metadata};
use crate::descriptor::descriptor_content::DescContent;
use core::marker::PhantomData;
use core::mem;
use rust_gpu_bindless_buffer_content::BufferStructPlain;
use spirv_std::ByteAddressableBuffer;

pub struct Buffer<T: BufferContent + ?Sized> {
	_phantom: PhantomData<T>,
}

impl<T: BufferContent + ?Sized> DescContent for Buffer<T> {}

pub struct MutBuffer<T: BufferContent + ?Sized> {
	_phantom: PhantomData<T>,
}

impl<T: BufferContent + ?Sized> DescContent for MutBuffer<T> {}

pub struct BufferSlice<'a, T: ?Sized> {
	buffer: ByteAddressableBuffer<&'a [u32]>,
	meta: Metadata,
	_phantom: PhantomData<T>,
}

pub struct MutBufferSlice<'a, T: ?Sized> {
	buffer: ByteAddressableBuffer<&'a mut [u32]>,
	meta: Metadata,
	_phantom: PhantomData<T>,
}

impl<'a, T: BufferContent + ?Sized> BufferSlice<'a, T> {
	/// # Safety
	/// T needs to match the contents of the buffer
	#[inline]
	pub unsafe fn from_slice(buffer: &'a [u32], meta: Metadata) -> Self {
		Self {
			buffer: ByteAddressableBuffer::from_slice(buffer),
			meta,
			_phantom: PhantomData {},
		}
	}
}

impl<'a, T: BufferContent + ?Sized> MutBufferSlice<'a, T> {
	/// # Safety
	/// T needs to match the contents of the buffer
	#[inline]
	pub unsafe fn from_mut_slice(buffer: &'a mut [u32], meta: Metadata) -> Self {
		Self {
			buffer: ByteAddressableBuffer::from_mut_slice(buffer),
			meta,
			_phantom: PhantomData {},
		}
	}

	pub fn as_ref(&self) -> BufferSlice<T> {
		BufferSlice {
			buffer: self.buffer.as_ref(),
			meta: self.meta,
			_phantom: self._phantom,
		}
	}
}

impl<'a, T: BufferStruct> BufferSlice<'a, T> {
	/// Loads a T from the buffer.
	pub fn load(&self) -> T {
		unsafe { T::read(self.buffer.load_unchecked(0), self.meta) }
	}
}

impl<'a, T: BufferStruct> MutBufferSlice<'a, T> {
	/// Loads a T from the buffer.
	pub fn load(&self) -> T {
		self.as_ref().load()
	}
}

impl<'a, T: BufferStructPlain> MutBufferSlice<'a, T> {
	/// Stores a T to the buffer.
	///
	/// # Safety
	/// Stores from different waves or invocations must not alias.
	/// Loading data written by another thread or invocation without a memory barrier in between is UB.
	pub unsafe fn store(&mut self, t: T) {
		self.buffer.store_unchecked(0, T::write(t));
	}
}

impl<'a, T: BufferStruct> BufferSlice<'a, [T]> {
	/// Loads a T at an `index` offset from the buffer.
	pub fn load(&self, index: usize) -> T {
		let byte_offset = index * mem::size_of::<T::Transfer>();
		unsafe { T::read(self.buffer.load(byte_offset as u32), self.meta) }
	}

	/// Loads a T at an `index` offset from the buffer.
	///
	/// # Safety
	/// `byte_index` must be in bounds of the buffer
	pub unsafe fn load_unchecked(&self, index: usize) -> T {
		let byte_offset = index * mem::size_of::<T::Transfer>();
		unsafe { T::read(self.buffer.load_unchecked(byte_offset as u32), self.meta) }
	}
}

impl<'a, T: BufferStruct> MutBufferSlice<'a, [T]> {
	/// Loads a T at an `index` offset from the buffer.
	pub fn load(&self, index: usize) -> T {
		self.as_ref().load(index)
	}

	/// Loads a T at an `index` offset from the buffer.
	///
	/// # Safety
	/// `byte_index` must be in bounds of the buffer.
	pub unsafe fn load_unchecked(&self, index: usize) -> T {
		self.as_ref().load_unchecked(index)
	}
}

impl<'a, T: BufferStructPlain> MutBufferSlice<'a, [T]> {
	/// Loads a T at an `index` offset from the buffer.
	///
	/// # Safety
	/// Stores from different waves or invocations must not alias.
	/// Loading data written by another thread or invocation without a memory barrier in between is UB.
	pub unsafe fn store(&mut self, index: usize, t: T) {
		let byte_offset = index * mem::size_of::<T::Transfer>();
		self.buffer.store(byte_offset as u32, T::write(t));
	}

	/// Loads a T at an `index` offset from the buffer.
	///
	/// # Safety
	/// Stores from different waves or invocations must not alias.
	/// Loading data written by another thread or invocation without a memory barrier in between is UB.
	/// `byte_index` must be in bounds of the buffer.
	pub unsafe fn store_unchecked(&mut self, index: usize, t: T) {
		let byte_offset = index * mem::size_of::<T::Transfer>();
		self.buffer.store_unchecked(byte_offset as u32, T::write(t));
	}
}

impl<'a, T: BufferContent + ?Sized> BufferSlice<'a, T> {
	/// Loads an arbitrary type E at an `byte_index` offset from the buffer. `byte_index` must be a multiple of 4,
	/// otherwise, it will get silently rounded down to the nearest multiple of 4.
	///
	/// # Safety
	/// E must be a valid arbitrary BufferStruct type located at that `byte_offset`.
	pub unsafe fn load_at_arbitrary_offset<E: BufferStruct>(&self, byte_offset: usize) -> E {
		unsafe { E::read(self.buffer.load(byte_offset as u32), self.meta) }
	}

	/// Loads an arbitrary type E at an `byte_index` offset from the buffer. `byte_index` must be a multiple of 4,
	/// otherwise, it will get silently rounded down to the nearest multiple of 4.
	///
	/// # Safety
	/// E must be a valid arbitrary BufferStruct type located at that `byte_offset`.
	/// `byte_index` must be in bounds of the buffer.
	pub unsafe fn load_at_arbitrary_offset_unchecked<E: BufferStruct>(&self, byte_offset: usize) -> E {
		unsafe { E::read(self.buffer.load_unchecked(byte_offset as u32), self.meta) }
	}
}

impl<'a, T: BufferContent + ?Sized> MutBufferSlice<'a, T> {
	/// Loads an arbitrary type E at an `byte_index` offset from the buffer. `byte_index` must be a multiple of 4,
	/// otherwise, it will get silently rounded down to the nearest multiple of 4.
	///
	/// # Safety
	/// E must be a valid arbitrary BufferStruct type located at that `byte_offset`.
	pub unsafe fn load_at_arbitrary_offset<E: BufferStruct>(&self, byte_offset: usize) -> E {
		self.as_ref().load_at_arbitrary_offset(byte_offset)
	}

	/// Loads an arbitrary type E at an `byte_index` offset from the buffer. `byte_index` must be a multiple of 4,
	/// otherwise, it will get silently rounded down to the nearest multiple of 4.
	///
	/// # Safety
	/// E must be a valid arbitrary BufferStruct type located at that `byte_offset`.
	/// `byte_index` must be in bounds of the buffer.
	pub unsafe fn load_at_arbitrary_offset_unchecked<E: BufferStruct>(&self, byte_offset: usize) -> E {
		self.as_ref().load_at_arbitrary_offset_unchecked(byte_offset)
	}
}

impl<'a, T: BufferContent + ?Sized> MutBufferSlice<'a, T> {
	/// Loads an arbitrary type E at an `byte_index` offset from the buffer. `byte_index` must be a multiple of 4,
	/// otherwise, it will get silently rounded down to the nearest multiple of 4.
	///
	/// # Safety
	/// E must be a valid arbitrary BufferStruct type located at that `byte_offset`.
	pub unsafe fn store_at_arbitrary_offset<E: BufferStructPlain>(&mut self, byte_offset: usize, e: E) {
		self.buffer.store_unchecked(byte_offset as u32, E::write(e));
	}

	/// Loads an arbitrary type E at an `byte_index` offset from the buffer. `byte_index` must be a multiple of 4,
	/// otherwise, it will get silently rounded down to the nearest multiple of 4.
	///
	/// # Safety
	/// E must be a valid arbitrary BufferStruct type located at that `byte_offset`.
	/// `byte_index` must be in bounds of the buffer.
	pub unsafe fn store_at_arbitrary_offset_unchecked<E: BufferStructPlain>(&mut self, byte_offset: usize, e: E) {
		self.buffer.store_unchecked(byte_offset as u32, E::write(e));
	}
}
