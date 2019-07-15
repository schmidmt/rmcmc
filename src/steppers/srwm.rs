use crate::{Parameter, StepperBuilder, SteppingAlg};
use rv::traits::Rv;
use serde::export::PhantomData;
use rand::Rng;

pub struct SRWM {}

impl SRWM {
    pub fn new() -> Self {
        Self {

        }
    }
}

impl<'a, Model, RNG> SteppingAlg<'a, Model, RNG> for SRWM
    where
        Model: Clone,
{
    fn step(&mut self, rng: &mut RNG, model: Model) -> Model {
        self.step_with_log_likelihood(rng, model, None).0
    }

    fn step_with_log_likelihood(&mut self, rng: &mut RNG, model: Model, log_likelihood: Option<f64>) -> (Model, f64) {
        return (model, 0.0)
    }

    fn draw_prior(&self, rng: &mut RNG, m: Model) -> Model {
        m
    }
}




#[derive(Clone)]
pub struct SRWMBuilder<'a, RV, Type, LogLikelihood, Model, RNG>
    where
        Type: Clone,
        RV: Rv<Type> + Clone + Sync + Send,
        LogLikelihood: Fn(&Model) -> f64 + Clone + Sync + Send,
        Model: Clone,
{
    log_likelihood: &'a LogLikelihood,
    parameter: &'a Parameter<RV, Type, Model>,
    phantom_data: PhantomData<RNG>,
}

impl<'a, RV, Type, LogLikelihood, Model, RNG> SRWMBuilder<'a, RV, Type, LogLikelihood, Model, RNG>
    where
        Type: Clone,
        RV: Rv<Type> + Clone + Sync + Send,
        LogLikelihood: Fn(&Model) -> f64 + Clone + Sync + Send,
        RNG: Rng + Clone + Sync + Send,
        Model: Clone,
{
    pub fn new(parameter: &'a Parameter<RV, Type, Model>, log_likelihood: &'a LogLikelihood) -> Self {
        Self {
            parameter,
            log_likelihood,
            phantom_data: PhantomData,
        }
    }

}


impl<'a, RV, Type, LogLikelihood, Model, RNG> StepperBuilder<'a, Model, RNG> for SRWMBuilder<'a, RV, Type, LogLikelihood, Model, RNG>
    where
        Model: Clone,
        Type: Clone,
        RV: Rv<Type> + Clone + Sync + Send,
        LogLikelihood: Fn(&Model) -> f64 + Clone + Sync + Send,
        RNG: Rng + Clone + Sync + Send
{
    fn build(&self) -> Box<dyn SteppingAlg<'a, Model, RNG> + 'a> {
        Box::new(SRWM::new())
    }
}
