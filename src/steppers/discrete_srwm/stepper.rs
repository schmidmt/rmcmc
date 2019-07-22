use rv::traits::Rv;
use rand::Rng;
use num::{Saturating, Integer, ToPrimitive, FromPrimitive};
use crate::{Parameter, SteppingAlg};
use crate::steppers::adaptors::{ScaleAdaptor, AdaptState};
use rv::dist::Geometric;
use crate::steppers::helpers::metropolis_proposal;
use crate::steppers::helpers::MHStatus::*;
use std::marker::PhantomData;

/// Extension Type for DiscreteSRWM supported types
pub trait DiscreteType: Integer + Saturating + ToPrimitive + FromPrimitive + Clone + Send + Sync  {}


/// Discrete Symmetric Random Walk Metropolis State
pub struct DiscreteSRWM<'a, Prior, Type, Model, LogLikelihood, Adaptor, RNG>
    where
        Type: DiscreteType,
        Prior: Rv<Type>,
        LogLikelihood: Fn(&Model) -> f64,
        RNG: Rng,
        Adaptor: ScaleAdaptor<Type>,
{
    parameter: &'a Parameter<Prior, Type, Model>,
    log_likelihood: &'a LogLikelihood,
    current_log_likelihood: Option<f64>,
    current_prior: Option<f64>,
    adaptor: Adaptor,
    phantom_rng: PhantomData<RNG>,
}


impl<'a, Prior, Type, Model, LogLikelihood, RNG, Adaptor> DiscreteSRWM<'a, Prior, Type, Model, LogLikelihood, Adaptor, RNG>
    where
        Type: DiscreteType,
        Prior: Rv<Type>,
        LogLikelihood: Fn(&Model) -> f64,
        RNG: Rng,
        Adaptor: ScaleAdaptor<Type>,
{
    /// Create a new DiscreteSRWM stepper
    ///
    /// # Parameters
    /// * `parameter` - Parameter to step.
    /// * `log_likelihood` - Log Likelihood function.
    /// * `adaptor` - Adaptor to scale proposals with.
    pub fn new(
        parameter: &'a Parameter<Prior, Type, Model>,
        log_likelihood: &'a LogLikelihood,
        adaptor: Adaptor,
    ) -> Self {
        Self {
            parameter,
            log_likelihood,
            current_log_likelihood: None,
            current_prior: None,
            adaptor,
            phantom_rng: PhantomData,
        }
    }
}

impl<'a, Prior, Type, Model, LogLikelihood, RNG, Adaptor> SteppingAlg<'a, Model, RNG> for DiscreteSRWM<'a, Prior, Type, Model, LogLikelihood, Adaptor, RNG>
    where
        Model: Clone,
        Type: DiscreteType,
        Prior: Rv<Type> + Send + Sync,
        LogLikelihood: Fn(&Model) -> f64 + Send + Sync,
        RNG: Rng + Send + Sync,
        Adaptor: ScaleAdaptor<Type>,
{
    fn step(&mut self, rng: &mut RNG, model: Model) -> Model {
        let current_ll = self.current_log_likelihood;
        self.step_with_log_likelihood(rng, model, current_ll).0
    }

    fn step_with_log_likelihood(&mut self, rng: &mut RNG, model: Model, log_likelihood: Option<f64>) -> (Model, f64) {

        // Current State
        let current_value = self.parameter.lens.get(&model);
        let current_ll = log_likelihood.unwrap_or_else(|| {
            (self.log_likelihood)(&model)
        });
        let current_prior = self.current_prior.unwrap_or_else(|| {
            self.parameter.prior.ln_f(&current_value)
        });
        let current_score = current_ll + current_prior;

        // Proposal Dist
        let scale2 = self.adaptor.scale().powi(2);
        let geom_p = ((4.0 * scale2 + 1.0).sqrt() - 1.0) / (2.0 * scale2);
        let propsal_dist = Geometric::new(geom_p).unwrap();

        // Absolute value of draw
        let mag: usize = propsal_dist.draw(rng);
        let mag: Type = Type::from_usize(mag).unwrap();

        // Determine proposed value draw after symmetrization
        let proposed_value = if rng.gen() {
            current_value.clone().saturating_add(mag)
        } else {
            current_value.clone().saturating_sub(mag)
        };

        let proposed_prior = self.parameter.prior.ln_f(&proposed_value);
        let proposed_model = self.parameter.lens.set(&model, proposed_value.clone());

        // If the prior score is infinite, we've likely moved out of it's support.
        // Continue with the infinite value to rejection.
        let mut proposed_ll: Option<f64> = None;
        let proposed_score = if proposed_prior.is_finite() {
            let ll = (self.log_likelihood)(&proposed_model);
            proposed_ll = Some(ll);
            ll + proposed_prior
        } else {
            proposed_prior
        };

        let log_alpha = proposed_score - current_score;

        let update = metropolis_proposal(
            rng,
            log_alpha,
            &proposed_value,
            current_value
        );

        self.adaptor.update(&update);
        match update {
            Accepted(_, _) => {
                self.current_log_likelihood = proposed_ll;
                self.current_prior = Some(proposed_prior);
                (proposed_model, log_likelihood.unwrap())
            },
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
