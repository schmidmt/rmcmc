//! # Stepping Algorithms
//!
//! The stepping algorithms are the engine of the rmcmc system.
//! Each stepper has the ability to update the state of the model.
//! Steppers can be grouped together with the Group stepper.
pub mod metropolis_hastings_utils;

pub mod adaptor;
//mod discrete_srwm;
// mod group;
pub mod srwm;
mod stepping_alg;

pub use self::adaptor::{AdaptationMode, AdaptationStatus};
// pub use self::discrete_srwm::DiscreteSRWM;
//pub use self::group::Group;
pub use self::srwm::SRWM;
pub use self::stepping_alg::{ModelAndLikelihood, StepperBuilder, SteppingAlg};
