//! An implementation of the Simple Scaling Adaptor

use std::marker::PhantomData;
use crate::steppers::adaptors::{Adaptor, AdaptState, ScaleAdaptor};
use crate::steppers::helpers::MHStatus;
use crate::steppers::helpers::MHStatus::{Accepted, Rejected};


/// # Simple Adaptor
/// A simple scale adaptor derived from
/// https://github.com/pymc-devs/pymc3/blob/4d1eb3f/pymc3/step_methods/metropolis.py#L180
#[derive(Clone, Debug)]
pub struct SimpleAdaptor<T>
where
    T: Clone,
{
    alpha_sum: f64,
    n_updates: usize,
    adapt_interval: usize,
    scale: f64,
    initial_scale: f64,
    enabled: bool,
    phantom_t: PhantomData<T>,
}

impl<T> SimpleAdaptor<T>
where
    T: Clone,
{
    /// Create a new simple adaptor that adapts every `adapt_interval` steps
    pub fn new(scale: f64, adapt_interval: usize) -> Self {
        Self {
            alpha_sum: 0.0,
            n_updates: 0,
            adapt_interval,
            scale,
            initial_scale: scale,
            enabled: false,
            phantom_t: PhantomData,
        }
    }
}

impl<T> Adaptor<T> for SimpleAdaptor<T>
where
    T: Clone + Send + Sync,
{
    fn update(&mut self, update: &MHStatus<T>) {
        if self.enabled {
            self.n_updates += 1;
            let alpha = match update {
                Accepted(_, a) => a.exp(),
                Rejected(_, a) => a.exp(),
            };
            self.alpha_sum += alpha;

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

    fn state(&self) -> AdaptState {
        if self.enabled {
            AdaptState::On
        } else {
            AdaptState::Off
        }
    }

    fn enable(&mut self) {
        self.enabled = true;
    }

    fn disable(&mut self) {
        self.enabled = false;
    }
}

impl<T> ScaleAdaptor<T, f64> for SimpleAdaptor<T>
    where
        T: Clone + Send + Sync,
{
    fn scale(&self) -> f64 {
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steppers::adaptors::{Adaptor, ScaleAdaptor};

    #[test]
    fn should_increase_scale_with_too_high_alpha() {
        let mut adaptor = SimpleAdaptor::new(1.0, 10);
        adaptor.enable();
        assert_eq!(adaptor.scale(), 1.0);

        for _ in 0..100 {
            adaptor.update(&Accepted(&1, 0.5));
        }
        assert!(adaptor.scale() > 1.0);
    }

    #[test]
    fn should_decrease_scale_with_too_low_alpha() {
        let mut adaptor = SimpleAdaptor::new(1.0, 10);
        adaptor.enable();
        assert_eq!(adaptor.scale(), 1.0);

        for _ in 0..100 {
            adaptor.update(&Accepted(&1, 0.1));
        }
        assert!(adaptor.scale() > 1.0);
    }

    #[test]
    fn should_return_expected_scaling() {
        let mut adaptor = SimpleAdaptor::new(1.0, 10);
        adaptor.enable();
        assert_eq!(adaptor.scale(), 1.0);

        for _ in 0..100 {
            adaptor.update(&Accepted(&1, 0.51_f64.ln()));
        }
        assert::close(adaptor.scale(), 1.1_f64.powi(10), 1E-12);
    }
}
