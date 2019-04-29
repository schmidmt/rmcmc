//! Symmetric Random Walk Metropolis

use std::fmt;
extern crate rand;
use rand::Rng;

use rv::dist::Gaussian;
use rv::traits::{Mean, Rv, Variance};

use parameter::Parameter;
use steppers::{SteppingAlg, AdaptationStatus, AdaptationMode, util, ModelAndLikelihood};
use steppers::adaptor::{ScaleAdaptor, GlobalAdaptor};

/// Symmetric Random Walk Metropolis Stepping Algorithm
#[derive(Clone)]
pub struct SRWM<D, T, V, M, L>
where
    D: Rv<T> + Clone + fmt::Debug,
    T: Copy + Clone + fmt::Debug,
    M: Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + fmt::Debug
{
    pub parameter: Parameter<D, T, M>,
    pub loglikelihood: L,
    current_loglikelihood_score: Option<f64>,
    current_prior_score: Option<f64>,
    adaptor: GlobalAdaptor<T, V>
}

impl<D, T, V, M, L> fmt::Debug for SRWM<D, T, V, M, L>
where
    D: Rv<T> + fmt::Debug + Clone,
    T: Clone + Copy + fmt::Debug,
    M: fmt::Debug + Clone,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + fmt::Debug
{ 
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SRWM {{ parameter: {:?}, loglikelihood: {:?}, log_prior: {:?}, adaptor: {:?} }}", self.parameter, self.current_loglikelihood_score, self.current_prior_score, self.adaptor)
    }
}

impl<D, T, V, M, L> SRWM<D, T, V, M, L>
where
    D: Rv<T> + Variance<V> + Mean<T> + Clone + fmt::Debug,
    T: Clone + Copy + fmt::Debug,
    M: Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + fmt::Debug + Copy
{
    pub fn new(
        parameter: Parameter<D, T, M>,
        loglikelihood: L,
        proposal_scale: Option<f64>,
    ) -> Option<Self> {
        let prior_variance = parameter.prior.variance()?;
        let prior_mean = parameter.prior.mean()?;

        let adaptor = GlobalAdaptor::new(
            proposal_scale.unwrap_or(1.0),
            prior_mean,
            prior_variance,
        );

        Some(SRWM {
            parameter,
            loglikelihood,
            current_loglikelihood_score: None,
            current_prior_score: None,
            adaptor,
        })
    }
}

impl<D, T, V, M, L> SRWM<D, T, V, M, L>
where
    D: Rv<T> + Clone + fmt::Debug,
    T: Clone + Copy + fmt::Debug,
    M: Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + fmt::Debug + Copy
{
    pub fn new_with_mean_variance(
        parameter: Parameter<D, T, M>,
        loglikelihood: L,
        proposal_scale: f64,
        prior_mean: T,
        prior_variance: V
    ) -> Option<Self> {
        let adaptor = GlobalAdaptor::new(
            proposal_scale,
            prior_mean,
            prior_variance,
        );

        Some(SRWM {
            parameter,
            loglikelihood,
            current_loglikelihood_score: None,
            current_prior_score: None,
            adaptor,
        })
    }
}

macro_rules! impl_float {
    ($type:ty) => {
        impl<D, M, L, R> SteppingAlg<M, R> for SRWM<D, $type, $type, M, L>
        where
            D: 'static + Rv<$type> + Clone + fmt::Debug,
            M: 'static + Clone + fmt::Debug,
            L: 'static + Fn(&M) -> f64 + Clone + Sync,
            R: 'static + Rng
        {
            fn set_adapt(&mut self, mode: AdaptationMode) {
                self.adaptor.set_mode(mode)
            }
        
            fn get_adapt(&self) -> AdaptationStatus {
                self.adaptor.get_mode()
            }
        
            fn reset(&mut self) {
                self.current_loglikelihood_score = None;
                self.current_prior_score = None;
                self.adaptor.reset();
            }
        
            fn step(&mut self, rng: &mut R, model: M) -> M {
                let current_loglikelihood = self.current_loglikelihood_score;
                self.step_with_loglikelihood(rng, model, current_loglikelihood).model
            }

            fn step_with_loglikelihood(&mut self, rng: &mut R, model: M, loglikelihood: Option<f64>) -> ModelAndLikelihood<M> {
                let current_value = self.parameter.lens.get(&model);
                let current_loglikelihood_score = loglikelihood
                    .unwrap_or_else(|| (self.loglikelihood)(&model));
                let current_prior_score = self.current_prior_score
                    .unwrap_or_else(|| self.parameter.prior.ln_f(&current_value));
                let current_score = current_loglikelihood_score + current_prior_score;

                // propose new value
                let proposal_dist = Gaussian::new(current_value as f64, self.adaptor.get_scale()).unwrap();
                let proposed_new_value: $type = proposal_dist.draw(rng);
                let new_model = self.parameter.lens.set(&model, proposed_new_value);
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
                let mut new_loglikelihood_score: Option<f64> = None;
                let new_score = if new_prior_score.is_finite() {
                    let ll = (self.loglikelihood)(&new_model);
                    new_loglikelihood_score = Some(ll);
                    ll + new_prior_score
                } else {
                    new_prior_score
                };
        
                let log_alpha = new_score - current_score;
                let update = util::metropolis_select(rng, log_alpha, proposed_new_value, current_value);
                self.adaptor.update(&update);
        
                match update { 
                    util::MetroplisUpdate::Accepted(_, _) => {
                        self.current_loglikelihood_score = new_loglikelihood_score;
                        self.current_prior_score = Some(new_prior_score);
                        ModelAndLikelihood::new(new_model, new_loglikelihood_score)
                    },
                    util::MetroplisUpdate::Rejected(_, _) => {
                        ModelAndLikelihood::new(model, Some(current_loglikelihood_score))
                    }
                }
            }
        
            fn box_clone(&self) -> Box<SteppingAlg<M, R>> {
                Box::new(self.clone())
            }

            fn prior_draw(&self, rng: &mut R, model: M) -> M {
                self.parameter.draw(&model, rng)
            }
        }
    };
}

impl_float!(f64);
impl_float!(f32);


#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;
    use lens::*;
    use runner::Runner;
    use rv::dist::*;
    use rv::misc::ks_test;
    use rv::prelude::Cdf;
    use utils::multiple_tries;
    use rand::SeedableRng;

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

        fn loglikelihood(m: &Model) -> f64 {
            Uniform::new(-1.0, 1.0).unwrap().ln_f(&m.x)
        }

        let alg_start = SRWM::new(parameter, loglikelihood, Some(0.7)).unwrap();

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };
            let runner = Runner::new(alg_start.clone())
                .warmup(0)
                .chains(1)
                .thinning(10);

            let results: Vec<Vec<Model>> = runner.run(&mut rng, m);

            let samples: Vec<f64> = results
                .iter()
                .map(|chain| -> Vec<f64> {
                    chain.iter().map(|g| g.x).collect()
                }).flatten()
                .collect();
            

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

        fn loglikelihood(m: &Model) -> f64 {
            Uniform::new(-1.0, 1.0).unwrap().ln_f(&m.x)
        }

        let alg_start = SRWM::new(parameter, loglikelihood, Some(0.7)).unwrap();

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };
            let results: Vec<Vec<Model>> =
                Runner::new(alg_start.clone())
                .thinning(10)
                .chains(1)
                .run(&mut rng, m);

            let samples: Vec<f64> = results
                .iter()
                .map(|chain| -> Vec<f64> {
                    chain.iter().map(|g| g.x).collect()
                }).flatten()
                .collect();

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

        let loglikelihood =
            |m: &Model| Gaussian::new(0.0, 1.0).unwrap().ln_f(&m.x);

        let alg_start = SRWM::new(parameter, loglikelihood.clone(), Some(0.7)).unwrap();

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };
            let results: Vec<Vec<Model>> =
                Runner::new(alg_start.clone())
                .thinning(10)
                .chains(1)
                .run(&mut rng, m);

            let samples: Vec<f64> = results
                .iter()
                .map(|chain| -> Vec<f64> {
                    chain.iter().map(|g| g.x).collect()
                }).flatten()
                .collect();

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
            let loglikelihood = move |m: &Model| {
                let dist = Gaussian::new(0.0, m.sigma2.sqrt()).unwrap();
                ll_data.iter().map(|x: &f64| -> f64 { dist.ln_f(x) }).sum()
            };

            let alg_start =
                SRWM::new(parameter, loglikelihood.clone(), Some(0.7))
                .expect("Failed to produce a new SRWM");

            let m = Model { sigma2: 1.0 };
            let results: Vec<Vec<Model>> = Runner::new(alg_start.clone())
                .thinning(100)
                .chains(2)
                .run(&mut rng, m);

            let samples: Vec<f64> = results
                .iter()
                .map(|chain| -> Vec<f64> {
                    chain.iter().map(|g| g.sigma2).collect()
                }).flatten()
                .collect();

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
