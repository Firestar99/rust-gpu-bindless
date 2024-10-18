// caused by AnyBitPattern derive on PushConstant
#![allow(clippy::multiple_bound_locations)]

use crate::buffer_content::{BufferContent, Metadata};
use crate::descriptor::image_types::standard_image_types;
use crate::descriptor::reference::{AliveDescRef, Desc};
use crate::descriptor::{Buffer, BufferSlice, DescContent};
use bytemuck_derive::AnyBitPattern;
use spirv_std::{RuntimeArray, Sampler, TypedBuffer};

/// Some struct that facilitates access to a [`ValidDesc`] pointing to some [`DescContent`]
pub trait DescriptorsAccess<C: DescContent + ?Sized> {
	fn access(&self, desc: &Desc<impl AliveDescRef, C>) -> C::AccessType<'_>;
}

macro_rules! decl_descriptors {
    (
		{$($storage_name:ident: $storage_ty:ty,)*}
		{$($sampled_name:ident: $sampled_ty:ty,)*}
	) => {
		pub struct Descriptors<'a> {
			pub buffers: &'a RuntimeArray<TypedBuffer<[u32]>>,
			$(pub $storage_name: &'a RuntimeArray<$storage_ty>,)*
			$(pub $sampled_name: &'a RuntimeArray<$sampled_ty>,)*
			pub samplers: &'a RuntimeArray<Sampler>,
			pub meta: Metadata,
		}
		$(
			impl<'a> DescriptorsAccess<$storage_ty> for Descriptors<'a> {
				fn access(&self, desc: &Desc<impl AliveDescRef, $storage_ty>) -> <$storage_ty as DescContent>::AccessType<'_> {
					unsafe { self.$storage_name.index(desc.id().index().to_usize()) }
				}
			}
		)*
		$(
			impl<'a> DescriptorsAccess<$sampled_ty> for Descriptors<'a> {
				fn access(&self, desc: &Desc<impl AliveDescRef, $sampled_ty>) -> <$sampled_ty as DescContent>::AccessType<'_> {
					unsafe { self.$sampled_name.index(desc.id().index().to_usize()) }
				}
			}
		)*
	};
}
standard_image_types!(decl_descriptors);

impl<'a, T: ?Sized + BufferContent + 'static> DescriptorsAccess<Buffer<T>> for Descriptors<'a> {
	fn access(&self, desc: &Desc<impl AliveDescRef, Buffer<T>>) -> <Buffer<T> as DescContent>::AccessType<'_> {
		unsafe { BufferSlice::new(self.buffers.index(desc.id().index().to_usize()), self.meta) }
	}
}

impl<'a> DescriptorsAccess<Sampler> for Descriptors<'a> {
	fn access(&self, desc: &Desc<impl AliveDescRef, Sampler>) -> <Sampler as DescContent>::AccessType<'_> {
		unsafe { self.samplers.index(desc.id().index().to_usize()) }
	}
}

/// All bindless push constants are this particular struct, with T being the declared push_param.
///
/// Must not derive `DescStruct`, as to [`DescStruct::from_transfer`] Self you'd need the Metadata, which this struct
/// contains. To break the loop, it just stores Metadata flat and params directly as `T::TransferDescStruct`.
#[repr(C)]
#[derive(Copy, Clone, AnyBitPattern)]
pub struct PushConstant<T: bytemuck::AnyBitPattern + Send + Sync> {
	pub t: T,
	pub metadata: Metadata,
}
