use rand::Rng;
use std::marker::PhantomData;
use steppers::{SteppingAlg, AdaptationStatus, AdaptationMode};
use reduce::Reduce;
use statistics::Statistic;
use std::fmt;

/// Stepper Group
pub struct Group<M, R: Rng>
where
    M: Clone,
{
    steppers: Vec<Box<(dyn SteppingAlg<M, R> + 'static)>>,
    phantom_m: PhantomData<M>,
}

impl<M, R: Rng> Group<M, R>
where
    M: Clone,
{
    pub fn new(steppers: Vec<Box<(dyn SteppingAlg<M, R> + 'static)>>) -> Self {
        Group {
            steppers: steppers,
            phantom_m: PhantomData,
        }
    }
}

impl<M, R> fmt::Debug for Group<M, R> 
where
    M: Clone + fmt::Debug,
    R: Rng
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Group {{ samplers: ")?;
        self.steppers.fmt(f)?;
        write!(f, " }}")
    }
}


impl<M, R: Rng> SteppingAlg<M, R> for Group<M, R>
where
    M: Clone + fmt::Debug,
{
    fn step(&mut self, rng: &mut R, model: M) -> M {
        self
            .steppers
            .iter_mut()
            .fold(model, |x, stepper| stepper.step(rng, x))
    }

    fn set_adapt(&mut self, mode: AdaptationMode) {
        self
            .steppers
            .iter_mut()
            .map(|s| s.set_adapt(mode))
            .collect()
    }

    fn get_adapt(&self) -> AdaptationStatus {
        self
            .steppers
            .iter()
            .map(|s| s.get_adapt())
            .reduce(|a, b| match (a, b) {
                (AdaptationStatus::Enabled, AdaptationStatus::Enabled) => AdaptationStatus::Enabled,
                (AdaptationStatus::Disabled, AdaptationStatus::Disabled) => AdaptationStatus::Disabled,
                _ => AdaptationStatus::Mixed
            })
            .unwrap_or(AdaptationStatus::Mixed)
    }

    fn get_statistics(&self) -> Vec<Statistic<M, R>> {
        self
            .steppers
            .iter()
            .flat_map(|s| s.get_statistics())
            .collect()
    }

    fn reset(&mut self) {
        self
            .steppers
            .iter_mut()
            .for_each(|s| s.reset())
    }
    
    /*
    fn substeppers(&self) -> Option<&Vec<Box<SteppingAlg<M, R>>>> {
        Some(&self.steppers)
    }

    fn prior_sample(&self, rand: &mut R, m: M) -> M {
        self
            .steppers
            .fold(m, |mp, s| { s.prior_sample(rand, mp) })
    }
    */
}
