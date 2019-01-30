//! # Stepping Algorithms
//!

use std::fmt::Debug;
use rand::Rng;
use statistics::Statistic;

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


/// A stepping algorithm which draws the next stage from the Markov Chain.
pub trait SteppingAlg<M, R: Rng>: Debug
{
    // Advance the parameters by one step.
    fn step(&mut self, rng: &mut R, model: M) -> M;
    // Set the adaptation mode
    fn set_adapt(&mut self, mode: AdaptationMode);
    // Enables adaption.
    // Return the adaptation status.
    fn get_adapt(&self) -> AdaptationStatus;
    // Return a list of statistics
    fn get_statistics(&self) -> Vec<Statistic<M, R>>;
    // Reset the current stepper to it's initial state
    fn reset(&mut self);
    /*
    // Return a list of sub steppers
    fn substeppers(&self) -> Option<&Vec<Box<SteppingAlg<M, R>>>>;
    // Sample from prior
    fn prior_sample(&self, &mut R, model: M) -> M;
    */
}

/*
 * /// A stepping algorithm which supports annealing
 * pub trait AnnealingAlg: SteppingAlg {
 *     // Sets the temperature for the next step operation.
 *     fn set_temperature(&self, t: f64) -> Self;
 * }
 */

mod group;
mod srwm;
mod mock;
// mod kameleon;

pub use self::group::Group;
pub use self::srwm::SRWM;
pub use self::mock::Mock;
// pub use self::kameleon::Kameleon;
