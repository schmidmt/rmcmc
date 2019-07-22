use crate::steppers::discrete_srwm::{DiscreteType, DiscreteSRWM};
use rv::traits::Rv;
use rand::Rng;
use crate::steppers::adaptors::SimpleAdaptor;
use crate::{Parameter, StepperBuilder, SteppingAlg};
use std::marker::PhantomData;

/// Builder state for a Discrete Symmetric Random Walk Metropolis
pub struct DiscreteSRWMBuilder<'a, Prior, Type, Model, LogLikelihood, RNG>
    where
        Type: DiscreteType,
        Prior: Rv<Type>,
        LogLikelihood: Fn(&Model) -> f64,
        RNG: Rng,
{
    parameter: &'a Parameter<Prior, Type, Model>,
    log_likelihood: &'a LogLikelihood,
    initial_scale: f64,
    adapt_interval: usize,
    phantom_rng: PhantomData<RNG>,
}

impl<'a, Prior, Type, Model, LogLikelihood, RNG> StepperBuilder<'a, Model, RNG> for DiscreteSRWMBuilder<'a, Prior, Type, Model, LogLikelihood, RNG>
    where
        Model: Clone + Send + Sync,
        Type: DiscreteType,
        Prior: Rv<Type> + Send + Sync,
        LogLikelihood: Fn(&Model) -> f64 + Send + Sync,
        RNG: Rng + Send + Sync + 'a,
{
    fn build(&self) -> Box<SteppingAlg<'a, Model, RNG> + 'a> {
        let adaptor = SimpleAdaptor::new(self.initial_scale, self.adapt_interval);
        Box::new(DiscreteSRWM::new(self.parameter, self.log_likelihood, adaptor))
    }
}
