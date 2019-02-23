//! Symmetric Random Walk Metropolis

use std::fmt;
extern crate rand;
use rand::Rng;

use rv::dist::{Gaussian, Geometric};
use rv::traits::{Mean, Rv, Variance};

use parameter::Parameter;
use steppers::{SteppingAlg, AdaptationStatus, AdaptationMode, util};
use statistics::Statistic;
use steppers::adaptor::{ScaleAdaptor, GlobalAdaptor};

pub trait RWT: fmt::Debug + Clone + Copy {}


/// Symmetric Random Walk Metropolis Stepping Algorithm
pub struct SRWM<D, T, V, M, L>
where
    D: Rv<T> + Variance<V> + Mean<T> + Clone + fmt::Debug,
    T: RWT,
    M: 'static + Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + fmt::Debug
{
    pub parameter: Parameter<D, T, M>,
    pub log_likelihood: L,
    pub current_score: Option<f64>,
    pub temperature: f64,
    pub log_acceptance: f64,
    adaptor: GlobalAdaptor<T, V>
}

impl <D, T, V, M, L> fmt::Debug for SRWM<D, T, V, M, L>
where
    D: Rv<T> + Variance<V> + Mean<T> + Clone + fmt::Debug,
    T: RWT,
    M: 'static + Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + fmt::Debug
{ 
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SRWM {{ parameter: {:?}, current_score: {:?}, adaptor: {:?} }}", self.parameter, self.current_score, self.adaptor)
    }
}


impl<D, T, V, M, L> SRWM<D, T, V, M, L>
where
    D: Rv<T> + Variance<V> + Mean<T> + Clone + fmt::Debug,
    T: RWT,
    M: 'static + Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
    V: Clone + fmt::Debug + Copy
{
    pub fn new(
        parameter: Parameter<D, T, M>,
        log_likelihood: L,
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
            log_likelihood,
            current_score: None,
            log_acceptance: 0.0,
            temperature: 1.0,
            adaptor: adaptor,
        })
    }
}

impl<D, T, V, M, L> Clone for SRWM<D, T, V, M, L>
where
        D: Rv<T> + Variance<V> + Mean<T> + Clone + fmt::Debug,
        T: RWT,
        M: 'static + Clone + fmt::Debug,
        L: Fn(&M) -> f64 + Clone + Sync,
        V: Clone + fmt::Debug
{
    fn clone(&self) -> Self {
        SRWM {
            parameter: self.parameter.clone(),
            log_likelihood: self.log_likelihood.clone(),
            current_score: self.current_score,
            log_acceptance: self.log_acceptance,
            adaptor: self.adaptor.clone(),
            temperature: 1.0
        }
    }
}


macro_rules! impl_traits_ordinal {
    ($dtype: ty, $vtype: ty) => {
        impl RWT for $dtype {}

        impl<D, M, L, R> SteppingAlg<M, R> for SRWM<D, $dtype, $vtype, M, L>
        where 
            D: Rv<$dtype> + Variance<$vtype> + Mean<$dtype> + Clone + fmt::Debug,
            M: 'static + Clone + fmt::Debug,
            L: Fn(&M) -> f64 + Clone + Sync + fmt::Debug,
            R: Rng
        {
            fn set_adapt(&mut self, mode: AdaptationMode) {
                self.adaptor.set_mode(mode);
            }

            fn get_adapt(&self) -> AdaptationStatus {
                self.adaptor.get_mode()
            }

            fn get_statistics(&self) -> Vec<Statistic<M, R>> {
                Vec::new()
            }

            fn reset(&mut self) {
                self.current_score = None;
                self.adaptor.reset();
            }

            /*
            fn substeppers(&self) -> Option<&Vec<Box<SteppingAlg<M, R>>>> {
                None
            }

            fn prior_sample(&self, rng: &mut R, m: M) -> M {
                self.parameter.draw(&m, &mut rng)
            }
            */

            fn step(&mut self, rng: &mut R, model: M) -> M {
                let current_value = self.parameter.lens.get(&model);
                let current_score = self.current_score.unwrap_or_else(|| {
                    (self.log_likelihood)(&model) + self.parameter.prior.ln_f(&current_value)
                });

                // propose new value
                let geom_p = ((4.0 * self.adaptor.proposal_scale * self.adaptor.proposal_scale + 1.0).sqrt() + 1.0) / (2.0 * self.adaptor.proposal_scale * self.adaptor.proposal_scale);
                let proposal_dist = Geometric::new(geom_p).unwrap();
                let mag: $dtype = proposal_dist.draw(rng);

                let proposed_new_value = if rng.gen() {
                    current_value + mag
                } else {
                    if mag > current_value {
                        0
                    } else {
                        current_value - mag
                    }
                };
                let new_model = self.parameter.lens.set(&model, proposed_new_value);
                let prior_score = self.parameter.prior.ln_f(&proposed_new_value);

                // If the prior score is infinite, we've likely moved out of it's support.
                // Continue with the infinite value to rejection.
                let new_score = if prior_score.is_finite() {
                    (self.log_likelihood)(&new_model) + prior_score
                } else {
                    prior_score
                };

                let log_alpha = new_score - current_score;

                let update = util::metropolis_select(rng, log_alpha, proposed_new_value, current_value);
                self.adaptor.update(&update);
                match update{
                    util::MetroplisUpdate::Accepted(_, _) => {
                        self.current_score = Some(new_score);
                        self.log_acceptance = log_alpha;
                        new_model
                    },
                    util::MetroplisUpdate::Rejected(_, _) => {
                        self.log_acceptance = log_alpha;
                        model
                    }
                }
            }
        }
    };
}

macro_rules! impl_traits_continuous {
    ($dtype: ty, $vtype: ty) => {
        
        impl RWT for $dtype {}


        impl<D, M, L, R> SteppingAlg<M, R> for SRWM<D, $dtype, $vtype, M, L>
        where
            D: Rv<$dtype> + Variance<$vtype> + Mean<$dtype> + Clone + fmt::Debug,
            M: 'static + Clone + fmt::Debug,
            L: Fn(&M) -> f64 + Clone + Sync,
            R: Rng
        {
            fn set_adapt(&mut self, mode: AdaptationMode) {
                self.adaptor.set_mode(mode)
            }

            fn get_adapt(&self) -> AdaptationStatus {
                self.adaptor.get_mode()
            }

            fn get_statistics(&self) -> Vec<Statistic<M, R>> {
                Vec::new()
            }

            fn reset(&mut self) {
                self.current_score = None;
                self.adaptor.reset();
            }

            /*
            fn substeppers(&self) -> Option<&Vec<Box<SteppingAlg<M, R>>>> {
                None
            }

            fn prior_sample(&self, rng: &mut R, m: M) -> M {
                self.parameter.draw(&m, &mut rng)
            }
            */

            fn step(&mut self, rng: &mut R, model: M) -> M {
                let current_value = self.parameter.lens.get(&model);
                let current_score = self.current_score.unwrap_or_else(|| {
                    (self.log_likelihood)(&model) + self.parameter.prior.ln_f(&current_value)
                });

                // propose new value
                let proposal_dist = Gaussian::new(f64::from(current_value), self.adaptor.proposal_scale).unwrap();

                let proposed_new_value = proposal_dist.draw(rng);
                let new_model = self.parameter.lens.set(&model, proposed_new_value);
                let prior_score = self.parameter.prior.ln_f(&proposed_new_value);

                // If the prior score is infinite, we've likely moved out of it's support.
                // Continue with the infinite value to rejection.
                let new_score = if prior_score.is_finite() {
                    (self.log_likelihood)(&new_model) + prior_score
                } else {
                    prior_score
                };

                let log_alpha = new_score - current_score;
                let update = util::metropolis_select(rng, log_alpha, proposed_new_value, current_value);
                self.adaptor.update(&update);

                match update { 
                    util::MetroplisUpdate::Accepted(_, _) => {
                        self.current_score = Some(new_score);
                        self.log_acceptance = log_alpha;
                        new_model
                    },
                    util::MetroplisUpdate::Rejected(_, _) => {
                        self.log_acceptance = log_alpha;
                        model
                    }
                }
            }
        }
    };
}

impl_traits_continuous!(f32, f32);
impl_traits_continuous!(f64, f64);

impl_traits_ordinal!(u16, f64);
impl_traits_ordinal!(u32, f64);


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
    const SEED: [u8; 32] = [0; 32];

    #[test]
    fn uniform_posterior_no_warmup() {
        #[derive(Copy, Clone, Debug)]
        struct Model {
            x: f64,
        }

        let mut rng = rand::rngs::StdRng::from_seed(SEED);

        let parameter = Parameter::new(
            "x".to_string(),
            Uniform::new(-1.0, 1.0).unwrap(),
            make_lens!(Model, f64, x),
        );

        fn log_likelihood(m: &Model) -> f64 {
            Uniform::new(-1.0, 1.0).unwrap().ln_f(&m.x)
        }

        let alg_start = SRWM::new(parameter, log_likelihood, Some(0.7)).unwrap();

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
            
            println!("{:?}", samples);


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

        let mut rng = rand::rngs::StdRng::from_seed(SEED);

        let parameter = Parameter::new(
            "x".to_string(),
            Uniform::new(-1.0, 1.0).unwrap(),
            make_lens!(Model, f64, x),
        );

        fn log_likelihood(m: &Model) -> f64 {
            Uniform::new(-1.0, 1.0).unwrap().ln_f(&m.x)
        }

        let alg_start = SRWM::new(parameter, log_likelihood, Some(0.7)).unwrap();

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

        let mut rng = rand::rngs::StdRng::from_seed(SEED);

        let log_likelihood =
            |m: &Model| Gaussian::new(0.0, 1.0).unwrap().ln_f(&m.x);

        let alg_start = SRWM::new(parameter, log_likelihood.clone(), Some(0.7)).unwrap();

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

        let mut rng = rand::rngs::StdRng::from_seed(SEED);

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
                // println!("Likelihood (sigma = {})", m.sigma);
                let dist = Gaussian::new(0.0, m.sigma2.sqrt()).unwrap();
                ll_data.iter().map(|x: &f64| -> f64 { dist.ln_f(x) }).sum()
            };

            let alg_start =
                SRWM::new(parameter, log_likelihood.clone(), Some(0.7))
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
