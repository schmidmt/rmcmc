//! Symmetric Random Walk Metropolis

use std::fmt;
extern crate rand;
use rand::Rng;

use rv::dist::{Gaussian, Geometric};
use rv::traits::{Mean, Rv, Variance};

use parameter::Parameter;
use traits::*;

/// SRWM Adaption Helper
#[derive(Clone, Copy, Debug)]
struct SrwmAdaptor {
    // Scale factor *λ*
    log_lambda: f64,
    // Stochastic Mean *μ*
    mu: f64,
    // Stochastic Scale *σ*
    sigma: f64,
    // Number of adaptation steps.
    step: usize,
    // Scale for proposals.
    pub proposal_scale: f64,
    // Alpha to stochastically optimize towards.
    target_alpha: f64,
    // Enables updates or not.
    enabled: bool,
}

impl SrwmAdaptor {
    pub fn new(initial_proposal_scale: f64, prior_mean: f64) -> Self {
        SrwmAdaptor {
            log_lambda: 0.0,
            mu: prior_mean,
            sigma: initial_proposal_scale,
            step: 0,
            proposal_scale: initial_proposal_scale,
            target_alpha: 0.234,
            enabled: false,
        }
    }

    pub fn update(&self, alpha: f64, new_value: f64) -> Self {
        if self.enabled {
            let g = 0.9 / ((self.step + 1) as f64).powf(0.9);
            let delta = new_value - self.mu;
            let bounded_alpha = alpha.min(1.0);
            let new_log_lambda =
                self.log_lambda + g * (bounded_alpha - self.target_alpha);
            let new_mu = self.mu + g * delta;
            let new_sigma = self.sigma + g * (delta * delta - self.sigma);
            let new_proposal_scale = new_log_lambda.exp() * new_sigma;

            assert!(
                new_proposal_scale > 0.0,
                format!(
                    "update SrwmAdaptor = {:?}  \
                        (new_lambda = {}, new_sigma = {})
                        with alpha = {}, \
                        new_value = {} \
                        caused a bad proposal_scale...",
                    self,
                    new_log_lambda.exp(),
                    new_sigma,
                    alpha,
                    new_value
                )
            );

            SrwmAdaptor {
                log_lambda: new_log_lambda,
                mu: new_mu,
                sigma: new_sigma,
                step: self.step + 1,
                proposal_scale: new_proposal_scale,
                ..*self
            }
        } else {
            *self
        }
    }

    pub fn enable(&self) -> Self {
        SrwmAdaptor {
            enabled: true,
            ..*self
        }
    }
    pub fn disable(&self) -> Self {
        SrwmAdaptor {
            enabled: false,
            ..*self
        }
    }
}

/// Symmetric Random Walk Metropolis Stepping Algorithm
pub struct SRWM<D, T, M, L>
where
    D: Rv<T> + Variance<f64> + Mean<f64> + Clone,
    M: Clone + 'static,
    T: fmt::Debug,
    L: Fn(&M) -> f64 + Sync + Clone,
{
    pub parameter: Parameter<D, T, M>,
    pub log_likelihood: L,
    pub current_score: Option<f64>,
    pub temperature: f64,
    pub log_acceptance: f64,
    adaptor: SrwmAdaptor,
}

impl<D, T, M, L> SRWM<D, T, M, L>
where
    D: Rv<T> + Variance<f64> + Mean<f64> + Clone,
    M: Clone,
    T: fmt::Debug,
    L: Fn(&M) -> f64 + Sync + Clone,
{
    pub fn new(
        parameter: Parameter<D, T, M>,
        log_likelihood: L,
        proposal_scale: Option<f64>,
    ) -> Self {
        let prior_variance = parameter.prior.variance().unwrap_or(1.0);
        let prior_mean = parameter.prior.mean().unwrap();

        let adaptor = SrwmAdaptor::new(
            proposal_scale.unwrap_or(prior_variance),
            prior_mean,
        );

        SRWM {
            parameter,
            log_likelihood,
            current_score: None,
            temperature: 1.0,
            log_acceptance: 0.0,
            adaptor,
        }
    }
}

macro_rules! impl_traits_generic {
    ($kind: ty) => {
        impl<D, M, L> Clone for SRWM<D, $kind, M, L>
        where
            D: Rv<$kind> + Variance<f64> + Mean<f64> + Clone,
            M: Clone,
            L: Fn(&M) -> f64 + Sync + Clone,
        {
            fn clone(&self) -> Self {
                SRWM {
                   parameter: self.parameter.clone(),
                   log_likelihood: self.log_likelihood.clone(),
                   current_score: self.current_score,
                   temperature: self.temperature,
                   log_acceptance: self.log_acceptance,
                   adaptor: self.adaptor,
                }
            }
        }
    };
}



macro_rules! impl_traits_ordinal {
    ($kind: ty) => {
        impl_traits_generic!($kind);

        impl<D, M, L> SteppingAlg<M> for SRWM<D, $kind, M, L>
        where 
            D: Rv<$kind> + Variance<f64> + Mean<f64> + Clone,
            M: Clone,
            L: Fn(&M) -> f64 + Sync + Clone,
        {
            fn step<R: Rng>(&self, rng: &mut R, model: &mut M) -> Self {
                let current_value = self.parameter.lens.get(model);
                let current_score = self.current_score.unwrap_or_else(|| {
                    (self.log_likelihood)(model) + self.parameter.prior.ln_f(&current_value)
                });

                // propose new value
                let geom_p = ((4.0 * self.adaptor.proposal_scale * self.adaptor.proposal_scale + 1.0).sqrt() + 1.0) / (2.0 * self.adaptor.proposal_scale * self.adaptor.proposal_scale);
                let proposal_dist = Geometric::new(geom_p).unwrap();
                let mag: $kind = proposal_dist.draw(rng);

                let proposed_new_value = if rng.gen() {
                    current_value + mag
                } else {
                    if mag > current_value {
                        0
                    } else {
                        current_value - mag
                    }
                };
                let new_model = self.parameter.lens.set(model, proposed_new_value);
                let prior_score = self.parameter.prior.ln_f(&proposed_new_value);

                // If the prior score is infinite, we've likely moved out of it's support.
                // Continue with the infinite value to rejection.
                let new_score = if prior_score.is_finite() {
                    (self.log_likelihood)(&new_model) + prior_score
                } else {
                    prior_score
                };

                let log_alpha = new_score - current_score;
                let log_u = rng.gen::<f64>().ln();

                if log_u * self.temperature < log_alpha {
                    let new_adaptor = self.adaptor.update(log_alpha.exp(), proposed_new_value as f64);
                    // Update Model
                    self.parameter.lens.set_in_place(model, proposed_new_value);

                    // println!("Accepted: {} -> {} ({})", current_value, proposed_new_value, log_alpha.exp());

                    // Return the updated model.
                    SRWM {
                        current_score: Some(new_score),
                        log_acceptance: log_alpha,
                        adaptor: new_adaptor,
                        ..(*self).clone()
                    }
                } else {
                    let new_adaptor = self.adaptor.update(log_alpha.exp(), f64::from(current_value));

                    // println!("Rejected: {} -> {} ({})", current_value, proposed_new_value, log_alpha.exp());

                    // Return the same model plus some changes to book-keeping
                    SRWM {
                        current_score: Some(current_score),
                        log_acceptance: log_alpha,
                        adaptor: new_adaptor,
                        ..(*self).clone()
                    }
                }
            }
            fn adapt_on(&self) -> Self { SRWM { adaptor: self.adaptor.enable(), ..(*self).clone()} }
            fn adapt_off(&self) -> Self { SRWM { adaptor: self.adaptor.disable(), ..(*self).clone()} }
        }
    };
}

macro_rules! impl_traits_continuous {
    ($kind: ty) => {
        impl_traits_generic!($kind);
        
        impl<D, M, L> SteppingAlg<M> for SRWM<D, $kind, M, L>
        where
            D: Rv<$kind> + Variance<f64> + Mean<f64> + Clone,
            M: Clone,
            L: Fn(&M) -> f64 + Sync + Clone,
        {
            fn step<R: Rng>(&self, rng: &mut R, model: &mut M) -> Self {
                let current_value = self.parameter.lens.get(model);
                let current_score = self.current_score.unwrap_or_else(|| {
                    (self.log_likelihood)(model) + self.parameter.prior.ln_f(&current_value)
                });

                // propose new value
                let proposal_dist = Gaussian::new(f64::from(current_value), self.adaptor.proposal_scale).unwrap();

                let proposed_new_value = proposal_dist.draw(rng);
                let new_model = self.parameter.lens.set(model, proposed_new_value);
                let prior_score = self.parameter.prior.ln_f(&proposed_new_value);

                // If the prior score is infinite, we've likely moved out of it's support.
                // Continue with the infinite value to rejection.
                let new_score = if prior_score.is_finite() {
                    (self.log_likelihood)(&new_model) + prior_score
                } else {
                    prior_score
                };

                let log_alpha = new_score - current_score;
                let log_u = rng.gen::<f64>().ln();

                if log_u * self.temperature < log_alpha {
                    let new_adaptor = self.adaptor.update(log_alpha.exp(), f64::from(proposed_new_value));
                    // Update Model
                    self.parameter.lens.set_in_place(model, proposed_new_value);

                    // println!("Accepted: {} -> {} ({})", current_value, proposed_new_value, log_alpha.exp());

                    // Return the updated model.
                    SRWM {
                        current_score: Some(new_score),
                        log_acceptance: log_alpha,
                        adaptor: new_adaptor,
                        ..(*self).clone()
                    }
                } else {
                    let new_adaptor = self.adaptor.update(log_alpha.exp(), f64::from(current_value));

                    // println!("Rejected: {} -> {} ({})", current_value, proposed_new_value, log_alpha.exp());

                    // Return the same model plus some changes to book-keeping
                    SRWM {
                        current_score: Some(current_score),
                        log_acceptance: log_alpha,
                        adaptor: new_adaptor,
                        ..(*self).clone()
                    }
                }
            }

            fn adapt_on(&self) -> Self { SRWM { adaptor: self.adaptor.enable(), ..(*self).clone()} }
            fn adapt_off(&self) -> Self { SRWM { adaptor: self.adaptor.disable(), ..(*self).clone()} }
        }

        impl<D, M, L> AnnealingAlg<M> for SRWM<D, $kind, M, L>
        where
            D: Rv<$kind> + Variance<f64> + Mean<f64> + Clone,
            M: Clone,
            L: Fn(&M) -> f64 + Sync + Clone,
        {
            fn set_temperature(&self, t: f64) -> Self {
                SRWM {
                    temperature: t,
                    ..(*self).clone()
                }
            }
        }
    };
}

impl_traits_continuous!(f32);
impl_traits_continuous!(f64);

impl_traits_ordinal!(u16);
impl_traits_ordinal!(u32);


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
    // use utils;
    // use std::path::Path;

    const P_VAL: f64 = 0.2;
    const N_TRIES: usize = 10;

    #[test]
    fn uniform_posterior_no_warmup() {
        #[derive(Copy, Clone, Debug)]
        struct Model {
            x: f64,
        }

        let parameter = Parameter::new(
            "x".to_string(),
            Uniform::new(-1.0, 1.0).unwrap(),
            make_lens!(Model, f64, x),
        );

        let log_likelihood =
            |m: &Model| Uniform::new(-1.0, 1.0).unwrap().ln_f(&m.x);

        let alg_start = SRWM::new(parameter, log_likelihood.clone(), Some(0.7));

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };
            let results: Vec<Vec<(Model, _)>> = Runner::new(alg_start.clone())
                .warmup(0)
                .chains(1)
                .thinning(10)
                .run(m);

            let samples: Vec<f64> = results
                .iter()
                .map(|chain| -> Vec<f64> {
                    chain.iter().map(|g| g.0.x).collect()
                }).flatten()
                .collect();

            let (stat, p) =
                ks_test(&samples, |s| Uniform::new(-1.0, 1.0).unwrap().cdf(&s));
            println!("test stat = {}, p = {}", stat, p);
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

        let parameter = Parameter::new(
            "x".to_string(),
            Uniform::new(-1.0, 1.0).unwrap(),
            make_lens!(Model, f64, x),
        );

        let log_likelihood =
            |m: &Model| Uniform::new(-1.0, 1.0).unwrap().ln_f(&m.x);

        let alg_start = SRWM::new(parameter, log_likelihood.clone(), Some(0.7));

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };
            let results: Vec<Vec<(Model, _)>> =
                Runner::new(alg_start.clone()).thinning(10).chains(1).run(m);

            let samples: Vec<f64> = results
                .iter()
                .map(|chain| -> Vec<f64> {
                    chain.iter().map(|g| g.0.x).collect()
                }).flatten()
                .collect();

            let (stat, p) =
                ks_test(&samples, |s| Uniform::new(-1.0, 1.0).unwrap().cdf(&s));
            println!("test stat = {}, p = {}", stat, p);
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

        let log_likelihood =
            |m: &Model| Gaussian::new(0.0, 1.0).unwrap().ln_f(&m.x);

        let alg_start = SRWM::new(parameter, log_likelihood.clone(), Some(0.7));

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0.0 };
            let results: Vec<Vec<(Model, _)>> =
                Runner::new(alg_start.clone()).thinning(10).chains(1).run(m);

            let samples: Vec<f64> = results
                .iter()
                .map(|chain| -> Vec<f64> {
                    chain.iter().map(|g| g.0.x).collect()
                }).flatten()
                .collect();

            let (stat, p) =
                ks_test(&samples, |s| Gaussian::new(0.0, 1.0).unwrap().cdf(&s));
            println!("test stat = {}, p = {}", stat, p);
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

        let mut rng = rand::thread_rng();

        let passed = multiple_tries(N_TRIES, |_| {
            let alpha = 2.0;
            let beta = 2.0;

            let parameter = Parameter::new(
                "sigma2".to_string(),
                InvGamma::new(alpha, beta).unwrap(),
                make_lens!(Model, f64, sigma2),
            );

            let data = Gaussian::new(0.0, 3.14).unwrap().sample(10, &mut rng);

            let ll_data = data.clone();
            let log_likelihood = move |m: &Model| {
                // println!("Likelihood (sigma = {})", m.sigma);
                let dist = Gaussian::new(0.0, m.sigma2.sqrt()).unwrap();
                ll_data.iter().map(|x: &f64| -> f64 { dist.ln_f(x) }).sum()
            };

            let alg_start =
                SRWM::new(parameter, log_likelihood.clone(), Some(0.7));

            let m = Model { sigma2: 1.0 };
            let results: Vec<Vec<(Model, _)>> =
                Runner::new(alg_start.clone()).thinning(100).chains(2).run(m);

            let samples: Vec<f64> = results
                .iter()
                .map(|chain| -> Vec<f64> {
                    chain.iter().map(|g| g.0.sigma2).collect()
                }).flatten()
                .collect();

            let new_alpha = alpha + data.len() as f64 / 2.0;
            let sum_of_squares: f64 = data.iter().map(|x| *x * *x).sum();
            let new_beta = beta + sum_of_squares / 2.0;

            println!("new_alpha = {}, new_beta = {}", new_alpha, new_beta);

            let expected_sigma_dist =
                InvGamma::new(new_alpha, new_beta).unwrap();

            // save samples to debug
            // utils::write_samples_to_file(Path::new(&format!("/tmp/samples_{}", i)), &samples).unwrap();

            let (stat, p) = ks_test(&samples, |s| expected_sigma_dist.cdf(&s));
            println!("test stat = {}, p = {}", stat, p);
            p > P_VAL
        });
        assert!(passed);
    }

}
