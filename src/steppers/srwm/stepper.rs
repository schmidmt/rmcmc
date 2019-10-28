use crate::steppers::adaptors::{
    AdaptState, Adaptor, GlobalAdaptor, ScaleAdaptor,
};
use crate::steppers::helpers::metropolis_proposal;
use crate::steppers::helpers::MHStatus::*;
use crate::{Parameter, SteppingAlg};
use num::Float;
use rand::Rng;
use rv::dist::Gaussian;
use rv::traits::Rv;
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

impl<'a, Prior, Type, Model, LogLikelihood, RNG> SteppingAlg<'a, Model, RNG>
    for SRWM<'a, Prior, Type, Type, Model, LogLikelihood, RNG>
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

    fn step_with_log_likelihood(
        &mut self,
        rng: &mut RNG,
        model: Model,
        log_likelihood: Option<f64>,
    ) -> (Model, f64) {
        // Determine current state
        let current_value = self.parameter.lens().get(&model);
        let current_ll =
            log_likelihood.unwrap_or_else(|| (self.log_likelihood)(&model));

        let current_prior = self
            .current_prior_score
            .unwrap_or_else(|| self.parameter.prior(&model).ln_f(&current_value));

        let current_score = current_ll + current_prior;

        // Start proposal
        assert!(self.adaptor.scale() > 0.0, "Cannot process scale <= 0");
        let proposal_dist =
            Gaussian::new(current_value.clone().into(), self.adaptor.scale())
                .unwrap();

        let proposed_value: f64 = proposal_dist.draw(rng);
        let proposed_value: Type = proposed_value.into();
        let proposed_model =
            self.parameter.lens().set(model.clone(), proposed_value);

        let proposed_prior = {
            let p = self.parameter.prior(&proposed_model).ln_f(&proposed_value);
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
            &current_value,
        );

        self.adaptor.update(&update);

        // Return appropriate value
        match update {
            Accepted(_, _) => {
                self.current_ll_score = proposed_ll;
                self.current_prior_score = Some(proposed_prior);
                (proposed_model, proposed_ll.unwrap())
            }
            Rejected(_, _) => (model, current_ll),
        }
    }
    fn draw_prior(&self, rng: &mut RNG, m: Model) -> Model {
        self.parameter.draw(m, rng)
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

#[cfg(test)]
mod tests {
    use log::info;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use crate::Lens;
    use crate::{make_lens, StepperBuilder};
    use crate::Parameter;
    use crate::steppers::srwm::SRWMBuilder;
    use crate::utils::test::assert_some_failures;
    use rv::dist::{Gaussian, Uniform};
    use rv::misc::ks_test;
    use rv::prelude::*;
    use std::sync::Mutex;

    const SEED: u64 = 0x726D636D63_u64;

    #[test]
    fn zero_likelihood_gaussian_prior() {
        let posterior = Gaussian::standard();
        let seed = Mutex::new(SEED);
        #[derive(Clone)]
        struct Model {
            x: f64,
        }

        assert_some_failures(5, || {
            let mut seed = seed.lock().unwrap();
            *seed += 1;
            let mut rng = StdRng::seed_from_u64(*seed);

            let log_likelihood = |_: &Model| { 0.0 };

            let x = Parameter::new_independent(
                Gaussian::standard(),
                make_lens!(Model, f64, x)
            );

            let stepper_builder = SRWMBuilder::new(
                &x,
                &log_likelihood,
                0.0,
                1.0
            );

            let mut stepper = stepper_builder.build();
            stepper.adapt_enable();
            stepper.multiple_steps(&mut rng, Model { x: 0.0 }, 1000);
            stepper.adapt_disable();

            let sample: Vec<f64> = stepper.sample(&mut rng, Model {x: 0.0}, 1000, 10)
                .iter()
                .map(|m| m.x)
                .collect();

            let (ks_stat, p_value) = ks_test(&sample, |x| posterior.cdf(&x));
            info!("KS: stat = {}, p-value = {}", ks_stat, p_value);
            assert!(p_value > 0.1)
        });
    }


    #[test]
    fn gaussian_likelihood_uniform_prior() {
        let posterior = Gaussian::standard();
        let seed = Mutex::new(SEED);
        #[derive(Clone)]
        struct Model {
            x: f64,
        }

        assert_some_failures(5, || {
            let mut seed = seed.lock().unwrap();
            *seed += 1;
            let mut rng = StdRng::seed_from_u64(*seed);

            let log_likelihood = |m: &Model| {
                Gaussian::standard().ln_f(&m.x)
            };

            let x = Parameter::new_independent(
                Uniform::new(-1000.0, 1000.0).unwrap(),
                make_lens!(Model, f64, x)
            );

            let stepper_builder = SRWMBuilder::new(
                &x,
                &log_likelihood,
                0.0,
                1.0
            );

            let mut stepper = stepper_builder.build();
            stepper.adapt_enable();
            stepper.multiple_steps(&mut rng, Model { x: 0.0 }, 1000);
            stepper.adapt_disable();

            let sample: Vec<f64> = stepper.sample(&mut rng, Model {x: 0.0}, 1000, 10)
                .iter()
                .map(|m| m.x)
                .collect();

            let (ks_stat, p_value) = ks_test(&sample, |x| posterior.cdf(&x));
            info!("KS: stat = {}, p-value = {}", ks_stat, p_value);
            assert!(p_value > 0.1)
        });
    }
}
