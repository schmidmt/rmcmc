//! An implementation of the Simple Scaling Adaptor

use steppers::adaptor::ScaleAdaptor;
use steppers::{AdaptationStatus, AdaptationMode};
use steppers::util::MetroplisUpdate;
use std::marker::PhantomData;

/// # Simple Adaptor
/// A simple scale adaptor derived from
/// https://github.com/pymc-devs/pymc3/blob/4d1eb3f/pymc3/step_methods/metropolis.py#L180
#[derive(Clone, Debug)]
pub struct SimpleAdaptor<T>
where
    T: Clone
{
    alpha_sum: f64,
    n_updates: usize,
    adapt_interval: usize,
    scale: f64,
    initial_scale: f64,
    enabled: bool,
    phantom_t: PhantomData<T>
}


impl<T> SimpleAdaptor<T>
where
    T: Clone,
{
    pub fn new(scale: f64, adapt_interval: usize) -> Self {
        Self {
            alpha_sum: 0.0,
            n_updates: 0,
            adapt_interval,
            scale,
            initial_scale: scale,
            enabled: false,
            phantom_t: PhantomData
        }
    }
}

impl<T> ScaleAdaptor<T> for SimpleAdaptor<T>
where
    T: Clone
{
    fn reset(&mut self) {
        self.alpha_sum = 0.0;
        self.n_updates = 0;
        self.scale = self.initial_scale;
    }

    fn get_scale(&self) -> f64 {
        self.scale
    }

    fn get_mode(&self) -> AdaptationStatus {
        match self.enabled {
            true => AdaptationStatus::Enabled,
            false  => AdaptationStatus::Disabled,
        }
    }

    fn set_mode(&mut self, mode: AdaptationMode) {
        match mode {
            AdaptationMode::Enabled => self.enabled = true,
            AdaptationMode::Disabled => self.enabled = false
        }
    }

    fn update(&mut self, update: &MetroplisUpdate<T>) {
        let alpha = match update {
            MetroplisUpdate::Accepted(_, a) => a,
            MetroplisUpdate::Rejected(_, a) => a,
        };
        self.alpha_sum += *alpha;

        if self.n_updates >= self.adapt_interval {
            let alpha_mean: f64 = self.alpha_sum / (self.n_updates as f64);
            if alpha_mean < 0.001 {
                self.scale *= 0.01;
            } else if alpha_mean < 0.05 {
                self.scale *= 0.5;
            } else if alpha_mean < 0.2 {
                self.scale *= 0.2;
            } else if alpha_mean > 0.95 {
                self.scale *= 10.0;
            } else if alpha_mean > 0.75 {
                self.scale *= 2.0;
            } else if alpha_mean > 0.5 {
                self.scale *= 1.1;
            }

            self.n_updates = 0;
            self.alpha_sum = 0.0;
        }
    }
}
