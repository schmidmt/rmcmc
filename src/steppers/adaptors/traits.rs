use crate::steppers::adaptors::AdaptState;
use crate::steppers::helpers::MHStatus;

/// General adaptor trait
pub trait Adaptor<Type> {
    /// Update the adaptor with an MHStatus
    fn update(&mut self, update: &MHStatus<Type>);
    /// Retrieve the current state
    fn state(&self) -> AdaptState;
    /// Enable updates to the adaptor
    fn enable(&mut self);
    /// Disable updates to the adaptor
    fn disable(&mut self);
}

/// An Adaptor with a adapted scale
pub trait ScaleAdaptor<Type>: Adaptor<Type> {
    /// Retrieve the current scale from the adaptor
    fn scale(&self) -> f64;
}
