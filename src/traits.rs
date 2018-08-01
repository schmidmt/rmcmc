extern crate rand;
use rand::Rng;

/// A stepping algorithm which draws the next stage from the Markov Chain.
pub trait SteppingAlg<M> {
    // Advance the parameters by one step.
    fn step<R: Rng>(&self, rng: &mut R, model: &mut M) -> Self;
    // Enables adaption.
    fn adapt_on(&self) -> Self;
    // Disables adaption.
    fn adapt_off(&self) -> Self;
}

/// A stepping algorithm which supports annealing
pub trait AnnealingAlg<M>: SteppingAlg<M> {
    // Sets the temperature for the next step operation.
    fn set_temperature(&self, t: f64) -> Self;
}
