//! RMCMC is a library for doing Markov-Chain Monte-Carlo for Rust.
//!
//! # Example
//! ```rust
//!
//! ```
// #![deny(missing_docs)]

#[macro_use]
mod lens;
pub use lens::*;

mod parameter;
pub use parameter::*;

mod stepper_traits;
pub use stepper_traits::*;


pub mod steppers;

mod runner;
pub use self::runner::Runner;

