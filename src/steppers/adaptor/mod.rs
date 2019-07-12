//! # Adaptors for hyper-parameter auto-tuning

use crate::steppers::metropolis_hastings_utils::MetropolisUpdate;

/// An adaptor that adapts the proposal scale
pub trait ScaleAdaptor<T>: Clone
where
    T: Clone,
{
    /// Update this adaptor with the given update action
    fn update(&mut self, update: &MetropolisUpdate<T>);
    /// Get the scale of the current adaptor
    fn scale(&self) -> f64;
    /// Set mode of the adaptor to `mode`
    fn set_mode(&mut self, mode: AdaptationMode);
    /// Get the current adaptor's mode
    fn mode(&self) -> AdaptationStatus;
    /// Reset to default inner values
    fn reset(&mut self);
}

mod global;
mod simple;
mod states;

pub use self::global::*;
pub use self::simple::*;
pub use self::states::*;
