use crate::{SteppingAlg, Parameter};
use rv::traits::Rv;
use rand::Rng;
use rv::dist::Gaussian;
use num::Float;
use crate::steppers::helpers::metropolis_proposal;
use crate::steppers::helpers::MHStatus::*;
use crate::steppers::adaptors::{GlobalAdaptor, ScaleAdaptor, Adaptor, AdaptState};
use std::marker::PhantomData;


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



impl<'a, Prior, Type, VType, Model, LogLikelihood, RNG> SRWM<'a, Prior, Type, VType, Model, LogLikelihood, RNG>
    where
        Prior: Rv<Type>,
        LogLikelihood: Fn(&Model) -> f64,
        RNG: Rng,
{
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


impl<'a, Prior, Type, Model, LogLikelihood, RNG> SteppingAlg<'a, Model, RNG> for SRWM<'a, Prior, Type, Type, Model, LogLikelihood, RNG>
    where
        Type: Clone + Float + Into<f64> + From<f64> + Send + Sync,
        Model: Clone + Send + Sync,
        Prior: Rv<Type> + Send + Sync,
        LogLikelihood: Fn(&Model) -> f64 + Send + Sync,
        RNG: Rng + Send + Sync,
{
    fn step(&mut self, rng: &mut RNG, model: Model) -> Model {
        let current_ll = self.current_ll_score;
        self.step_with_log_likelihood(rng, model, current_ll).0
    }

    fn step_with_log_likelihood(&mut self, rng: &mut RNG, model: Model, log_likelihood: Option<f64>) -> (Model, f64) {
        // Determine current state
        let current_value = self.parameter.lens.get(&model);
        let current_ll = log_likelihood
            .unwrap_or_else(|| (self.log_likelihood)(&model));

        let current_prior = self.current_prior_score
            .unwrap_or_else(|| self.parameter.prior.ln_f(&current_value));

        let current_score = current_ll + current_prior;

        // Start proposal
        assert!(self.adaptor.scale() > 0.0, "Cannot process scale <= 0");
        let proposal_dist = Gaussian::new(
            current_value.clone().into(),
            self.adaptor.scale()
        ).unwrap();

        let proposed_value: f64 = proposal_dist.draw(rng);
        let proposed_value: Type = proposed_value.into();
        let proposed_model = self.parameter.lens.set(&model, proposed_value.clone());

        let proposed_prior = {
            let p = self.parameter.prior.ln_f(&proposed_value);
            if p.is_nan() {
                std::f64::NEG_INFINITY
            } else {
                p
            }
        };

        let mut proposed_ll: Option<f64> = None;

        let proposed_score = if proposed_prior.is_finite() {
            let ll = (self.log_likelihood)(&proposed_model);
            proposed_ll = Some(ll);
            ll + proposed_prior
        } else {
            proposed_prior
        };

        // Do Metropolis Step

        let log_alpha = proposed_score - current_score;
        let update = metropolis_proposal(
            rng,
            log_alpha,
            &proposed_value,
            &current_value
        );

        self.adaptor.update(&update);


        // Return appropriate value
        match update {
            Accepted(_, _) => {
                self.current_ll_score = proposed_ll;
                self.current_prior_score = Some(proposed_prior);
                (proposed_model, proposed_ll.unwrap())
            }
            Rejected(_, _) => {
                (model, current_ll)
            }
        }
    }
    fn draw_prior(&self, rng: &mut RNG, m: Model) -> Model {
        self.parameter.draw(&m, rng)
    }

    fn adapt_enable(&mut self) {
        self.adaptor.enable();
    }

    fn adapt_disable(&mut self) {
        self.adaptor.disable();
    }

    fn adapt_state(&self) -> AdaptState {
        self.adaptor.state()
    }
}
