use crate::steppers::*;
use rand::Rng;
use reduce::Reduce;
use std::marker::PhantomData;

/// Stepper Group
pub struct Group<'a, M, R>
where
    M: Clone,
    R: Rng,
{
    /// Inner steppers for group
    pub steppers: Vec<Box<dyn SteppingAlg<'a, M, R> + 'a>>,
    current_likelihood: Option<f64>,
    phantom_m: PhantomData<M>,
    phantom_r: PhantomData<R>,
}

impl<'a, M, R> Clone for Group<'a, M, R>
where
    M: Clone,
    R: Rng,
{
    fn clone(&self) -> Self {
        let steppers = self.steppers.clone();

        Self {
            steppers,
            current_likelihood: self.current_likelihood,
            phantom_m: PhantomData,
            phantom_r: PhantomData,
        }
    }
}

unsafe impl<'a, M, R> Sync for Group<'a, M, R>
where
    M: Clone,
    R: Rng,
{
}

unsafe impl<'a, M, R> Send for Group<'a, M, R>
where
    M: Clone,
    R: Rng,
{
}

impl<'a, M, R> Group<'a, M, R>
where
    M: Clone,
    R: Rng,
{
    /// Create a new group with inner stepers `steppers`
    pub fn new(steppers: Vec<Box<dyn SteppingAlg<'a, M, R> + 'a>>) -> Self {
        Group {
            steppers,

            current_likelihood: None,
            phantom_m: PhantomData,
            phantom_r: PhantomData,
        }
    }
}

impl<'a, M, R> SteppingAlg<'a, M, R> for Group<'a, M, R>
where
    M: 'static + Clone,
    R: 'static + Rng,
{
    fn step(&mut self, rng: &mut R, model: M) -> M {
        let current_log_likelihood = self.current_likelihood;
        self.step_with_log_likelihood(rng, model, current_log_likelihood)
            .model
    }

    fn step_with_log_likelihood(
        &mut self,
        rng: &mut R,
        model: M,
        log_likelihood: Option<f64>,
    ) -> ModelAndLikelihood<M> {
        let next_mll = self.steppers.iter_mut().fold(
            ModelAndLikelihood::new(model, log_likelihood),
            |mll, stepper| {
                stepper.step_with_log_likelihood(
                    rng,
                    mll.model,
                    mll.log_likelihood,
                )
            },
        );
        self.current_likelihood = next_mll.log_likelihood;
        next_mll
    }

    fn set_adapt(&mut self, mode: AdaptationMode) {
        self.steppers.iter_mut().for_each(|s| s.set_adapt(mode))
    }

    fn adapt(&self) -> AdaptationStatus {
        self.steppers
            .iter()
            .map(|s| s.adapt())
            .reduce(|a, b| match (a, b) {
                (AdaptationStatus::Enabled, AdaptationStatus::Enabled) => {
                    AdaptationStatus::Enabled
                }
                (AdaptationStatus::Disabled, AdaptationStatus::Disabled) => {
                    AdaptationStatus::Disabled
                }
                _ => AdaptationStatus::Mixed,
            })
            .unwrap_or(AdaptationStatus::Mixed)
    }

    fn reset(&mut self) {
        self.steppers.iter_mut().for_each(|s| s.reset())
    }

    fn box_clone(&self) -> Box<dyn SteppingAlg<'a, M, R> + 'a> {
        Box::new((*self).clone())
    }

    fn prior_draw(&self, rng: &mut R, model: M) -> M {
        self.steppers
            .iter()
            .fold(model, |acc, s| s.prior_draw(rng, acc))
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    // use super::*;
    use nalgebra::inverse;
    use nalgebra::{DMatrix, DVector, Matrix2, Vector, Vector2};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use reduce::Reduce;
    use rv::dist::{Gaussian, MvGaussian};
    use rv::misc::ks_test;
    use rv::prelude::*;
    use std::rc::Rc;
    use double::*;

    use crate::lens::*;
    use crate::parameter::Parameter;
    use crate::runner::Runner;
    use crate::steppers::{SRWM, ModelAndLikelihood, AdaptationMode, AdaptationStatus, SteppingAlg};
    use rand::rngs::mock::StepRng;


    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Model1D {
        i: i32,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Model2D(DVector<f64>);

    #[derive(Clone)]
    struct Model {}

    impl Default for Model {
        fn default() -> Self {
            Model {}
        }
    }
}
