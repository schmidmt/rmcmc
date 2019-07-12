//! RMCMC is a library for doing Markov-Chain Monte-Carlo for Rust.
//!
//! # Example
//! ```rust
//!
//! ```
#![deny(missing_docs)]

#[macro_use]
mod lens;
pub use lens::*;

mod parameter;
pub use parameter::*;

mod runner;
pub use runner::*;

pub mod diagnostics;
pub mod likelihood;
pub mod steppers;
pub mod utils;
