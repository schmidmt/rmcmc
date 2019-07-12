/// Adaptation status query result
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AdaptationStatus {
    /// The stepper is adapting
    Enabled,
    /// The stepper is not adapting
    Disabled,
    /// The stepper's adaptation state is mixed or unknown.
    Mixed,
}

impl Default for AdaptationStatus {
    fn default() -> Self {
        AdaptationStatus::Mixed
    }
}

/// Imperative adaptation modes
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AdaptationMode {
    /// Enable adaptation on the target stepper
    Enabled,
    /// Disable adaptation on the target stepper
    Disabled,
}
