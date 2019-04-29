use steppers::util::MetroplisUpdate;

pub trait ScaleAdaptor<T>: Clone
where
    T: Clone
{
    fn update(&mut self, update: &MetroplisUpdate<T>);
    fn get_scale(&self) -> f64;
    fn set_mode(&mut self, mode: AdaptationMode);
    fn get_mode(&self) -> AdaptationStatus;
    fn reset(&mut self);
}

mod global;
mod simple;
mod states;

pub use self::states::*;
pub use self::simple::*;
pub use self::global::*;
