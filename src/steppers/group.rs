use rand::Rng;
use std::marker::PhantomData;
use steppers::*;
use reduce::Reduce;
use std::fmt;

/// Stepper Group
pub struct Group<M, R: Rng>
where
    M: 'static + Clone,
    R: 'static + Rng,
{
    steppers: Vec<Box<SteppingAlg<M, R>>>,
    current_likelihood: Option<f64>,
    phantom_m: PhantomData<M>,
    phantom_r: PhantomData<R>
}

impl<M, R> Clone for Group<M, R>
where
    M: 'static + Clone,
    R: 'static + Rng + Clone,
{
    fn clone(&self) -> Self {
        Self {
            steppers: self.steppers.iter().map(|x| (*x).box_clone()).collect(),
            current_likelihood: self.current_likelihood,
            phantom_m: PhantomData,
            phantom_r: PhantomData,
        }
    }
}

unsafe impl<M, R> Sync for Group<M, R>
where
    M: Clone,
    R: Rng,
{}

unsafe impl<M, R> Send for Group<M, R>
where
    M: Clone,
    R: Rng,
{}


impl<M, R: Rng> Group<M, R>
where
    M: Clone,
    R: Rng,
{
    pub fn new(steppers: Vec<Box<SteppingAlg<M, R>>>) -> Self {
        Group {
            steppers,
            current_likelihood: None,
            phantom_m: PhantomData,
            phantom_r: PhantomData
        }
    }
}

impl<M, R> fmt::Debug for Group<M, R> 
where
    M: Clone,
    R: Rng,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Group {{ samplers: ")?;
        self.steppers.fmt(f)?;
        write!(f, " }}")
    }
}


impl<M, R> SteppingAlg<M, R> for Group<M, R>
where
    M: Clone + fmt::Debug,
    R: Rng + Clone,
{
    fn step(&mut self, rng: &mut R, model: M) -> M {
        self
            .steppers
            .iter_mut()
            .fold(ModelWithScore::new(model, self.current_likelihood))(|


        self
            .steppers
            .iter_mut()
            .fold(model, |x, stepper| stepper.step(rng, x))
    }

    fn step_with_score(&mut self, rng: &mut R, model_with_score: ModelWithScore<M>) -> ModelWithScore<M> {
        self.steppers
            .iter_mut()
            .fold(model_with_score, |x, stepper| stepper.step_with_score(rng, x))
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

    fn reset(&mut self) {
        self
            .steppers
            .iter_mut()
            .for_each(|s| s.reset())
    } 

    fn box_clone(&self) -> Box<SteppingAlg<M, R>> {
        Box::new((*self).clone())
    }

    fn prior_draw(&self, rng: &mut R, model: M) -> M {
        self.steppers.iter().fold(model, |acc, s| s.prior_draw(rng, acc))
    }
}
