
#[derive(Clone, Copy, PartialEq, Hash)]
pub enum AdaptState {
    On,
    Off,
    Mixed,
    NotApplicable,
}

impl AdaptState {
    pub fn merge(&self, other: AdaptState) -> AdaptState {
        use AdaptState::*;
        match (self, other) {
            (NotApplicable, x) => x,
            (x, NotApplicable) => x.clone(),
            (On, On) => On,
            (Off, Off) => Off,
            _ => Mixed
        }
    }
}