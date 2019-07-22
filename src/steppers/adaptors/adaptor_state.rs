
/// The adaptation state of an adaptor or stepper.
#[derive(Clone, Copy, PartialEq, Hash)]
pub enum AdaptState {
    /// Adaptation is enabled.
    On,
    /// Adaptation is disabled.
    Off,
    /// Some components have adaptation enabled and disabled.
    Mixed,
    /// The current adaptor or stepping algorithm has no relevant adaptation state.
    NotApplicable,
    /// Something is preventing us from knowing the adaptation state.
    Unknown,
}

impl AdaptState {

    /// Merge this `AdaptState` with another.
    ///
    /// This is useful for processing groups of steppers.
    pub fn merge(self, other: AdaptState) -> AdaptState {
        use AdaptState::*;
        match (self, other) {
            (Unknown, _) => Unknown,
            (_, Unknown) => Unknown,
            (NotApplicable, x) => x,
            (x, NotApplicable) => x,
            (On, On) => On,
            (Off, Off) => Off,
            _ => Mixed,
        }
    }
}

impl Default for AdaptState {
    fn default() -> Self {
        AdaptState::Unknown
    }
}
