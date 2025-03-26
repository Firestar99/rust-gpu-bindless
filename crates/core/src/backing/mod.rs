//! Backend mod is the backing data structure for managing slots in the global descriptor sets and their delayed
//! destruction.

pub mod ab;
pub mod range_set;
pub mod slot_array;
pub mod table;

/// nothing uses the Platform generic, just reexport
#[cfg(feature = "primary")]
pub(crate) mod primary {
	pub use crate::backing::*;
}
