#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AdaptationStatus {
    Enabled,
    Disabled,
    Mixed
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AdaptationMode {
    Enabled,
    Disabled
}
