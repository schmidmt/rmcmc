//! # Stepping Algorithms
//!

use std::fmt::Debug;
use rand::Rng;

pub mod util;

#[derive(Copy, Clone, Debug)]
pub enum AdaptationStatus {
    Enabled,
    Disabled,
    Mixed
}

#[derive(Copy, Clone, Debug)]
pub enum AdaptationMode {
    Enabled,
    Disabled
}

pub struct ModelWithScore<M> {
    model: M,
    score: Option<f64>
}

impl<M> ModelWithScore<M> {
    pub fn new(model: M, score: f64) -> Self {
        Self { model, score: Some(score) }
    }

    pub fn new_no_score(model: M) -> Self {
        Self { model, score: None }
    }
}



/// A stepping algorithm which draws the next stage from the Markov Chain.
pub trait SteppingAlg<M, R>: Debug
where
    M: 'static,
    R: 'static + Rng
{
    // Advance the parameters by one step.
    fn step(&mut self, rng: &mut R, model: M) -> M;
    fn step_with_score(&mut self, rng: &mut R, model_with_score: ModelWithScore<M>) -> ModelWithScore<M>;
    // Set the adaptation mode
    fn set_adapt(&mut self, mode: AdaptationMode);
    // Enables adaption.
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
    R: 'static + Rng,
    M: 'static
{
    fn clone(&self) -> Box<SteppingAlg<M, R>> {
        self.box_clone()
    }
}

pub mod adaptor;
mod group;
mod srwm;
mod discrete_srwm;
mod binary_metropolis;
mod mock;

pub use self::group::Group;
pub use self::srwm::SRWM;
pub use self::discrete_srwm::DiscreteSRWM;
pub use self::mock::Mock;
pub use self::binary_metropolis::BinaryMetropolis;
