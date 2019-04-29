use rand::Rng;
use super::{AdaptationStatus, AdaptationMode};

pub struct ModelAndLikelihood<M> {
    pub model: M,
    pub loglikelihood: Option<f64>,
}

impl<M> ModelAndLikelihood<M> {
    pub fn new(model: M, loglikelihood: Option<f64>) -> Self {
        Self { model, loglikelihood }
    }
}

/// A stepping algorithm which draws the next stage from the Markov Chain.
pub trait SteppingAlg<M, R>
where
    M: 'static,
    R: 'static + Rng
{
    // Advance the parameters by one step.
    fn step(&mut self, rng: &mut R, model: M) -> M;
    // Advance by one step but with a given log_likelihood. (Helpful to prevent unnecessary
    // computations)
    fn step_with_loglikelihood(&mut self, rng: &mut R, model: M, loglikelihood: Option<f64>) -> ModelAndLikelihood<M>;
    // Set the adaptation mode
    fn set_adapt(&mut self, mode: AdaptationMode);
    // Return the adaptation status.
    fn get_adapt(&self) -> AdaptationStatus;
    // Reset the current stepper to it's initial state
    fn reset(&mut self);
    // Clone into a Box
    fn box_clone(&self) -> Box<SteppingAlg<M, R>>;
    // Draw from Prior
    fn prior_draw(&self, rng: &mut R, model: M) -> M;
}

impl<M, R> Clone for Box<SteppingAlg<M, R>>
where
    M: 'static,
    R: 'static + Rng
{
    fn clone(&self) -> Box<SteppingAlg<M, R>> {
        self.box_clone()
    }
}
