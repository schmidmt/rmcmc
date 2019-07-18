use crate::steppers::srwm::SRWM;
use crate::{StepperBuilder, SteppingAlg, Parameter};
use rv::traits::Rv;
use rand::Rng;
use std::marker::PhantomData;
use crate::steppers::adaptors::GlobalAdaptor;
use num::Float;

#[derive(Clone)]
pub struct SRWMBuilder<'a, RV, Type, VType, LogLikelihood, Model, RNG>
    where
        Type: Clone,
        VType: Clone,
        RV: Rv<Type> + Clone + Sync + Send,
        LogLikelihood: Fn(&Model) -> f64 + Clone + Sync + Send,
        Model: Clone,
{
    log_likelihood: &'a LogLikelihood,
    parameter: &'a Parameter<RV, Type, Model>,
    phantom_data: PhantomData<RNG>,
    initial_proposal_mean: Type,
    initial_proposal_variance: VType,
    initial_scale: f64,
}

impl<'a, RV, Type, VType, LogLikelihood, Model, RNG> SRWMBuilder<'a, RV, Type, VType, LogLikelihood, Model, RNG>
    where
        Type: Clone,
        VType: Clone,
        RV: Rv<Type> + Clone + Sync + Send,
        LogLikelihood: Fn(&Model) -> f64 + Clone + Sync + Send,
        RNG: Rng + Clone + Sync + Send,
        Model: Clone,
{
    pub fn new(
        parameter: &'a Parameter<RV, Type, Model>,
        log_likelihood: &'a LogLikelihood,
        initial_proposal_mean: Type,
        initial_proposal_variance: VType,
    ) -> Self {
        Self {
            parameter,
            log_likelihood,
            initial_proposal_mean,
            initial_proposal_variance,
            phantom_data: PhantomData,
            initial_scale: 1.0
        }
    }

    pub fn initial_scale(&self, initial_scale: f64) -> Self {
        assert!(initial_scale > 0.0, "The scale must be greater than zero");
        Self {
            initial_scale,
            ..self.clone()
        }
    }

}
impl<'a, RV, Type, LogLikelihood, Model, RNG> StepperBuilder<'a, Model, RNG> for SRWMBuilder<'a, RV, Type, Type, LogLikelihood, Model, RNG>
    where
        Model: Clone + Send + Sync,
        Type: Float + From<f64> + Into<f64> + Clone + Send + Sync,
        RV: Rv<Type> + Clone + Sync + Send,
        LogLikelihood: Fn(&Model) -> f64 + Clone + Sync + Send,
        RNG: 'a + Rng + Clone + Sync + Send
{
    fn build(&self) -> Box<dyn SteppingAlg<'a, Model, RNG> + 'a> {
        let adaptor = GlobalAdaptor::new(
            self.initial_scale,
            self.initial_proposal_mean.clone(),
            self.initial_proposal_variance.clone()
        );
        Box::new(SRWM::new(
            self.parameter,
            self.log_likelihood,
            adaptor,
        ))
    }
}
