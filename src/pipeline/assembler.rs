//! Assembler strategies: the `Assembler` trait and concrete implementations.
pub mod simple;
pub mod traits;

pub use simple::SimpleAssembler;
pub use traits::Assembler;