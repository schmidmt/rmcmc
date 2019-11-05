use std::marker::PhantomData;

use rv::traits::Rv;
use rand::Rng;

use crate::traits::*;
use crate::steppers::adaptors::SimpleAdaptor;
use crate::{Parameter, StepperBuilder, SteppingAlg};
use crate::steppers::discrete_srwm::DiscreteSRWM;

/// Builder state for a Discrete Symmetric Random Walk Metropolis
#[derive(Clone)]
pub struct DiscreteSRWMBuilder<'a, Prior, Type, Model, LogLikelihood, RNG>
    where
        Model: Clone,
        Type: DiscreteType,
        Prior: Rv<Type> + Clone,
        LogLikelihood: Fn(&Model) -> f64 + Clone,
        RNG: Rng + Clone,
{
    parameter: &'a Parameter<Prior, Type, Model>,
    log_likelihood: &'a LogLikelihood,
    initial_scale: f64,
    adapt_interval: usize,
    phantom_rng: PhantomData<RNG>,
}

impl<'a, Prior, Type, Model, LogLikelihood, RNG> DiscreteSRWMBuilder<'a, Prior, Type, Model, LogLikelihood, RNG>
    where
        Model: Clone,
        Type: DiscreteType,
        Prior: Rv<Type> + Clone,
        LogLikelihood: Fn(&Model) -> f64 + Clone,
        RNG: Rng + Clone,
{
    /// Create a new DiscreteSRWM Builder
    ///
    /// # Parameters
    /// * `parameter` - Parameter to update .
    /// * `log_likelihood` - Log Likelihood function.
    pub fn new(parameter: &'a Parameter<Prior, Type, Model>, log_likelihood: &'a LogLikelihood) -> Self {
        Self {
            parameter,
            log_likelihood,
            initial_scale: 1.0,
            adapt_interval: 100,
            phantom_rng: PhantomData,
        }
    }

    /// Set the initial proposal scale
    pub fn initial_scale(self, initial_scale: f64) -> Self {
        Self {
            initial_scale,
            ..self
        }
    }

    /// Set the adapt interval for the `SimpleAdaptor`.
    pub fn adapt_interval(self, adapt_interval: usize) -> Self {
        Self {
            adapt_interval,
            ..self
        }
    }
}


impl<'a, Prior, Type, Model, LogLikelihood, RNG> StepperBuilder<'a, Model, RNG> for DiscreteSRWMBuilder<'a, Prior, Type, Model, LogLikelihood, RNG>
    where
        Model: Clone + Send + Sync,
        Type: DiscreteType,
        Prior: Rv<Type> + Send + Sync + Clone,
        LogLikelihood: Fn(&Model) -> f64 + Send + Sync + Clone,
        RNG: Rng + Send + Sync + Clone + 'a,
{
    fn build(&self) -> Box<dyn SteppingAlg<'a, Model, RNG> + 'a> {
        let adaptor = SimpleAdaptor::new(self.initial_scale, self.adapt_interval);
        Box::new(DiscreteSRWM::new(self.parameter, self.log_likelihood, adaptor))
    }
}
