use rand::Rng;
use std::fmt;

use rv::dist::Gaussian;
use rv::traits::Rv;

use crate::parameter::Parameter;
use crate::steppers::adaptor::{GlobalAdaptor, ScaleAdaptor};
use crate::steppers::{
    metropolis_hastings_utils, AdaptationMode, AdaptationStatus,
    ModelAndLikelihood, SteppingAlg,
};
use num_traits::Float;

/// Symmetric Random Walk Metropolis Stepping Algorithm
pub struct SRWM<'a, D, T, V, M, L, R>
where
    D: Rv<T> + Clone,
    T: Clone,
    M: Clone,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone,
    R: Rng,
{
    /// parameter for which this stepper updates.
    parameter: &'a Parameter<D, T, M>,
    /// The log_likelihood function.
    log_likelihood: &'a L,
    /// Likelihood score for most recent model.
    current_log_likelihood_score: Option<f64>,
    /// Current prior score for most recent model.
    current_prior_score: Option<f64>,
    /// Adaptor used to adapt hyper-parameters.
    adaptor: GlobalAdaptor<T, V>,
    /// Random Number generator
    rng: &'a mut R,
}

impl<'a, D, T, V, M, L, R> fmt::Debug for SRWM<'a, D, T, V, M, L, R>
where
    D: Rv<T> + fmt::Debug + Clone,
    T: Clone + Copy + fmt::Debug,
    M: fmt::Debug + Clone,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + fmt::Debug,
    R: Rng,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SRWM {{ parameter: {:?}, log_likelihood: {:?}, log_prior: {:?}, adaptor: {:?} }}", self.parameter, self.current_log_likelihood_score, self.current_prior_score, self.adaptor)
    }
}

impl<'a, D, T, V, M, L, R> SRWM<'a, D, T, V, M, L, R>
where
    D: Rv<T> + Clone,
    T: Clone + Copy,
    M: Clone,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + Copy,
    R: Rng,
{
    /// Returns new SRWM stepper
    ///
    /// # Arguments
    /// * `parameter` - Parameter to step over with this stepper.
    /// * `log_likelihood` - Function to calculate the likelihood.
    /// * `proposal_scale` - Optional scale to begin proposals.
    ///
    /// # Example
    /// ```rust
    /// use rmcmc::{Parameter, Lens, make_lens};
    /// use rmcmc::steppers::SRWM;
    /// use rv::dist::Gaussian;
    /// use rv::traits::Rv;
    /// use rand::rngs::mock::StepRng;
    /// use rmcmc::steppers::adaptor::GlobalAdaptor;
    ///
    /// #[derive(Clone, Copy, Debug)]
    /// struct Model {
    ///     x: f64
    /// }
    ///
    /// let parameter = Parameter::new("x".to_owned(), Gaussian::new(0.0, 1.0).unwrap(), make_lens!(Model, f64, x));
    /// let log_likelihood = |m: &Model| {
    ///     let g = Gaussian::standard();
    ///     g.ln_f(&m.x)
    /// };
    ///
    /// let mut rng = StepRng::new(0, 1);
    ///
    /// let adaptor = GlobalAdaptor::new(0.0, 0.0, 1.0);
    ///
    /// let stepper = SRWM::new(&parameter, &log_likelihood, &mut rng, adaptor);
    /// ```
    pub fn new(
        parameter: &'a Parameter<D, T, M>,
        log_likelihood: &'a L,
        rng: &'a mut R,
        adaptor: GlobalAdaptor<T, V>,
    ) -> Self {
        Self {
            parameter,
            log_likelihood,
            current_log_likelihood_score: None,
            current_prior_score: None,
            adaptor,
            rng,
        }
    }
}

impl<'a, D, M, L, R, T> SteppingAlg<M> for SRWM<'a, D, T, T, M, L, R>
where
    D: 'static + Rv<T> + Clone + Sync + Send,
    M: 'static + Clone,
    L: 'static + Fn(&M) -> f64 + Clone + Sync,
    R: 'static + Rng + Sync + Send,
    T: 'static + Float + Into<f64> + From<f64> + Sync + Send,
{
    fn step(&mut self, model: M) -> M {
        let current_log_likelihood = self.current_log_likelihood_score;
        self.step_with_log_likelihood(model, current_log_likelihood)
            .model
    }

    fn step_with_log_likelihood(
        &mut self,
        model: M,
        log_likelihood: Option<f64>,
    ) -> ModelAndLikelihood<M> {
        let current_value = self.parameter.lens.get(&model);
        let current_log_likelihood_score =
            log_likelihood.unwrap_or_else(|| (self.log_likelihood)(&model));
        let current_prior_score = self
            .current_prior_score
            .unwrap_or_else(|| self.parameter.prior.ln_f(&current_value));
        let current_score = current_log_likelihood_score + current_prior_score;

        // propose new value
        let proposal_dist =
            Gaussian::new(current_value.clone().into(), self.adaptor.scale())
                .unwrap();
        let proposal_new_value_f: f64 = proposal_dist.draw(self.rng);
        let proposed_new_value: T = proposal_new_value_f.into();
        let new_model =
            self.parameter.lens.set(&model, proposed_new_value.clone());

        // Calculate the prior-score
        let new_prior_score = {
            let p = self.parameter.prior.ln_f(&proposed_new_value);
            if p.is_nan() {
                std::f64::NEG_INFINITY
            } else {
                p
            }
        };

        // If the prior score is infinite, we've likely moved out of it's support.
        // Continue with the infinite value to rejection.
        let mut new_log_likelihood_score: Option<f64> = None;

        let new_score = if new_prior_score.is_finite() {
            let ll = (self.log_likelihood)(&new_model);
            new_log_likelihood_score = Some(ll.clone());
            ll + new_prior_score
        } else {
            new_prior_score
        };

        let log_alpha = new_score - current_score;
        let update = metropolis_hastings_utils::metropolis_select(
            self.rng,
            log_alpha,
            &proposed_new_value,
            current_value,
        );
        self.adaptor.update(&update);

        match update {
            metropolis_hastings_utils::MetropolisUpdate::Accepted(_, _) => {
                self.current_log_likelihood_score = new_log_likelihood_score;
                self.current_prior_score = Some(new_prior_score);
                ModelAndLikelihood::new(new_model, new_log_likelihood_score)
            }
            metropolis_hastings_utils::MetropolisUpdate::Rejected(_, _) => {
                ModelAndLikelihood::new(
                    model,
                    Some(current_log_likelihood_score),
                )
            }
        }
    }

    fn set_adapt(&mut self, mode: AdaptationMode) {
        self.adaptor.set_mode(mode)
    }

    fn adapt(&self) -> AdaptationStatus {
        self.adaptor.mode()
    }

    fn prior_draw(&mut self, model: M) -> M {
        self.parameter.draw(&model, self.rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rv::dist::*;
    use rv::misc::ks_test;
    use rv::prelude::Cdf;

    use crate::lens::*;
    use crate::runner::draw_from_stepper;
    use crate::utils::multiple_tries;

    const P_VAL: f64 = 0.2;
    const N_TRIES: usize = 10;

    #[test]
    fn uniform_posterior_no_warmup() {
        #[derive(Copy, Clone, Debug)]
        struct Model {
            x: f64,
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let parameter = Parameter::new(
            "x".to_string(),
            Uniform::new(-1.0, 1.0).unwrap(),
            make_lens!(Model, f64, x),
        );

        fn log_likelihood(m: &Model) -> f64 {
            Uniform::new(-1.0, 1.0).unwrap().ln_f(&m.x)
        }

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };
            let adaptor = GlobalAdaptor::new(0.0, 0.0, 1.0);

            let alg_start =
                SRWM::new(&parameter, &log_likelihood, &mut rng, adaptor);

            let results: Vec<Model> =
                draw_from_stepper(alg_start, m, 2000, 1000, 10, false);

            let samples: Vec<f64> = results.iter().map(|g| g.x).collect();

            let (_stat, p) =
                ks_test(&samples, |s| Uniform::new(-1.0, 1.0).unwrap().cdf(&s));
            p > P_VAL
        });
        assert!(passed);
    }

    #[test]
    fn uniform_posterior_warmup() {
        #[derive(Copy, Clone, Debug)]
        struct Model {
            x: f64,
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let parameter = Parameter::new(
            "x".to_string(),
            Uniform::new(-1.0, 1.0).unwrap(),
            make_lens!(Model, f64, x),
        );

        fn log_likelihood(m: &Model) -> f64 {
            Uniform::new(-1.0, 1.0).unwrap().ln_f(&m.x)
        }

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };

            let adaptor = GlobalAdaptor::new(0.0, 0.0, 1.0);

            let alg_start =
                SRWM::new(&parameter, &log_likelihood, &mut rng, adaptor);

            let results: Vec<Model> =
                draw_from_stepper(alg_start, m, 2000, 1000, 10, false);

            let samples: Vec<f64> = results.iter().map(|g| g.x).collect();

            let (_stat, p) =
                ks_test(&samples, |s| Uniform::new(-1.0, 1.0).unwrap().cdf(&s));
            p > P_VAL
        });
        assert!(passed);
    }

    #[test]
    fn gaussian_likelihood_uniform_prior() {
        #[derive(Copy, Clone, Debug)]
        struct Model {
            x: f64,
        }

        let parameter = Parameter::new(
            "x".to_string(),
            Uniform::new(-10.0, 10.0).unwrap(),
            make_lens!(Model, f64, x),
        );

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let log_likelihood =
            |m: &Model| Gaussian::new(0.0, 1.0).unwrap().ln_f(&m.x);



        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };

            let adaptor = GlobalAdaptor::new(0.0, 0.0, 1.0);
            let alg_start =
                SRWM::new(&parameter, &log_likelihood, &mut rng, adaptor);

            let results: Vec<Model> =
                draw_from_stepper(alg_start, m, 2000, 1000, 10, false);

            let samples: Vec<f64> = results.iter().map(|g| g.x).collect();

            let (_stat, p) =
                ks_test(&samples, |s| Gaussian::new(0.0, 1.0).unwrap().cdf(&s));
            p > P_VAL
        });
        assert!(passed);
    }

    #[test]
    fn gaussian_fit() {
        #[derive(Copy, Clone, Debug)]
        struct Model {
            sigma2: f64,
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let passed = multiple_tries(N_TRIES, |_| {
            let alpha = 3.0;
            let beta = 1.0;

            let parameter = Parameter::new(
                "sigma2".to_string(),
                InvGamma::new(alpha, beta).unwrap(),
                make_lens!(Model, f64, sigma2),
            );

            let data = Gaussian::new(0.0, 3.14).unwrap().sample(10, &mut rng);

            let ll_data = data.clone();
            let log_likelihood = move |m: &Model| {
                let dist = Gaussian::new(0.0, m.sigma2.sqrt()).unwrap();
                ll_data.iter().map(|x: &f64| -> f64 { dist.ln_f(x) }).sum()
            };

            let adaptor = GlobalAdaptor::new(0.0, 1.0, 1.0);

            let alg_start =
                SRWM::new(&parameter, &log_likelihood, &mut rng, adaptor);

            let m = Model { sigma2: 1.0 };

            let results: Vec<Model> =
                draw_from_stepper(alg_start, m, 2000, 1000, 10, false);

            let samples: Vec<f64> = results.iter().map(|g| g.sigma2).collect();

            let new_alpha = alpha + data.len() as f64 / 2.0;
            let sum_of_squares: f64 = data.iter().map(|x| *x * *x).sum();
            let new_beta = beta + sum_of_squares / 2.0;

            let expected_sigma_dist =
                InvGamma::new(new_alpha, new_beta).unwrap();

            // save samples to debug
            // utils::write_samples_to_file(Path::new(&format!("/tmp/samples_{}", i)), &samples).unwrap();

            let (_stat, p) = ks_test(&samples, |s| expected_sigma_dist.cdf(&s));
            p > P_VAL
        });
        assert!(passed);
    }
}
