use rand::Rng;
use rv::dist::Gaussian;
use rv::traits::Rv;

use log::debug;

use crate::traits::*;
use crate::steppers::adaptors::{AdaptState, ScaleAdaptor, Adaptor};
use crate::steppers::helpers::metropolis_proposal;
use crate::steppers::helpers::MHStatus::*;
use crate::SteppingAlg;
use super::*;


impl<'a, Prior, Type, Model, LogLikelihood, RNG> SteppingAlg<'a, Model, RNG>
    for SRWM<'a, Prior, Type, Type, Model, LogLikelihood, RNG>
where
    Type: ScalarType,
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
        assert!(self.adaptor.scale().to_f64().unwrap() > 0.0, "Cannot process scale <= 0");
        let proposal_dist =
            Gaussian::new(current_value.clone().into(), self.adaptor.scale().to_f64().unwrap())
                .unwrap();

        let proposed_value: f64 = proposal_dist.draw(rng);
        let proposed_value: Type = Type::from_f64(proposed_value).unwrap();
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

        debug!("Prior = {}", proposed_prior);

        let mut proposed_ll: Option<f64> = None;

        let proposed_score = if proposed_prior.is_finite() {
            let mut ll = (self.log_likelihood)(&proposed_model);
            if ll.is_nan() {
                ll = std::f64::NEG_INFINITY;
            }
            proposed_ll = Some(ll);
            ll + proposed_prior
        } else {
            proposed_prior
        };

        debug!("Proposed LL = {:?}", proposed_ll);

        // Do Metropolis Step
        let log_alpha = proposed_score - current_score;
        let update = metropolis_proposal(
            rng,
            log_alpha,
            &proposed_value,
            &current_value,
        );
        
        debug!("Metropolis Step: {:?}", update);

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
    use crate::geweke::*;
    use crate::Lens;
    use crate::{make_lens, StepperBuilder};
    use crate::Parameter;
    use crate::steppers::srwm::SRWMBuilder;
    use crate::utils::test::assert_some_failures;
    use log::info;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
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

    #[test]
    fn geweke() {
        #[derive(Clone)]
        struct Model {
            mu: f64,
            sigma2: f64,
            data: Vec<f64>,
        };

        let mu_parameter = Parameter::new_independent(
            Gaussian::default(),
            make_lens!(Model, mu)
        );


        let init_model: Model = Model {
            mu: 0.0,
            sigma2: 1.0,
            data: vec![0.0],
        };

        fn loglikelihood(model: &Model) -> f64 {
            let g = Gaussian::new(model.mu, model.sigma2.sqrt()).expect("Bad parameters for Normal");
            model.data.iter().fold(0.0, |acc, x| acc + g.ln_f(x))
        }

        let builder = SRWMBuilder::new(
            &mu_parameter,
            &loglikelihood,
            0.0,
            0.2
        );

        fn to_stats(model: &Model) -> Vec<f64> {
            // vec![model.mu, model.sigma2.sqrt()]
            vec![model.mu]
        }

        fn resample_data(model: Model, rng: &mut StdRng) -> Model {
            let g = Gaussian::new(model.mu, model.sigma2.sqrt()).expect("Bad parameters for Normal");
            Model {
                data: g.sample(10, rng),
                ..model
            }
        }

        let mut rng = StdRng::seed_from_u64(0x1234);
        let config = GewekeConfig::new(500, 200, 10, 0.05, false);

        assert!(geweke_test(
            config,
            builder,
            init_model,
            to_stats,
            resample_data,
            &mut rng
        ));
    }
}
