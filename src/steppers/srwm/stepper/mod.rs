use crate::steppers::adaptors::GlobalAdaptor;
use crate::Parameter;
use rand::Rng;
use rv::traits::Rv;
use std::marker::PhantomData;

mod scalar;
pub use scalar::*;

mod vector;
pub use vector::*;

/// Symmetric Random Walk Metropolis
/// A random walk stepper with Gaussian proposals
///
/// # Types
///
pub struct SRWM<'a, Prior, Type, VType, Model, LogLikelihood, RNG>
where
    Prior: Rv<Type>,
    LogLikelihood: Fn(&Model) -> f64,
    RNG: Rng,
{
    parameter: &'a Parameter<Prior, Type, Model>,
    log_likelihood: &'a LogLikelihood,
    current_ll_score: Option<f64>,
    current_prior_score: Option<f64>,
    adaptor: GlobalAdaptor<Type, VType>,
    phantom_rng: PhantomData<RNG>,
}

impl<'a, Prior, Type, VType, Model, LogLikelihood, RNG>
    SRWM<'a, Prior, Type, VType, Model, LogLikelihood, RNG>
where
    Prior: Rv<Type>,
    LogLikelihood: Fn(&Model) -> f64,
    RNG: Rng,
{
    /// Create a new SRWM stepper
    ///
    /// # Parameter
    /// * `parameter` - Parameter updated by this stepper.
    /// * `log_likelihood` - Log Likelihood.
    /// * `adaptor` - Adaptor to use to dynamically update the proposal scale.
    pub fn new(
        parameter: &'a Parameter<Prior, Type, Model>,
        log_likelihood: &'a LogLikelihood,
        adaptor: GlobalAdaptor<Type, VType>,
    ) -> Self {
        Self {
            parameter,
            log_likelihood,
            current_ll_score: None,
            current_prior_score: None,
            adaptor,
            phantom_rng: PhantomData,
        }
    }
}

