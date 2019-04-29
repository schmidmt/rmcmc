//! Symmetric Random Walk Metropolis for Discrete RVs

use std::fmt;

use rand::Rng;
use rv::dist::Geometric;
use rv::traits::Rv;

use crate::parameter::Parameter;
use crate::steppers::{SteppingAlg, AdaptationStatus, AdaptationMode, util, ModelAndLikelihood};
use crate::steppers::adaptor::{SimpleAdaptor, ScaleAdaptor};

use num::{Integer, Saturating, ToPrimitive, FromPrimitive};


/// Symmetric Random Walk Metropolis Stepping Algorithm
#[derive(Clone)]
pub struct DiscreteSRWM<T, D, M, L>
where
    T: Integer + Saturating + ToPrimitive + FromPrimitive + Clone + fmt::Debug,
    D: Rv<T> + Clone + fmt::Debug,
    M: Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
{
    pub parameter: Parameter<D, T, M>,
    loglikelihood: L,
    current_loglikelihood_score: Option<f64>,
    current_prior_score: Option<f64>,
    adaptor: SimpleAdaptor<T>,
    step: usize,
}
impl<T, D, M, L> fmt::Debug for DiscreteSRWM<T, D, M, L>
where
    T: Integer + Saturating + ToPrimitive + FromPrimitive + Clone + fmt::Debug,
    D: Rv<T> + Clone + fmt::Debug,
    M: Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
{ 
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DiscreteSRWM {{ step: {} parameter: {:?}, loglikelihood: {:?}, logprior: {:?} adaptor: {:?} }}", self.step, self.parameter, self.current_loglikelihood_score, self.current_prior_score, self.adaptor)
    }
}

impl<T, D, M, L> DiscreteSRWM<T, D, M, L>
where
    T: Integer + Saturating + ToPrimitive + FromPrimitive + Clone + fmt::Debug,
    D: Rv<T> + Clone + fmt::Debug,
    M: Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
{
    pub fn new(
        parameter: Parameter<D, T, M>,
        loglikelihood: L,
        proposal_scale: Option<f64>,
    ) -> Option<Self> {
        let adaptor = SimpleAdaptor::new(
            proposal_scale.unwrap_or(1.0),
            100
        );

        Some(DiscreteSRWM {
            parameter,
            loglikelihood,
            current_loglikelihood_score: None,
            current_prior_score: None,
            adaptor,
            step: 0,
        })
    }
}

impl<T, D, M, L, R> SteppingAlg<M, R> for DiscreteSRWM<T, D, M, L>
where 
    T: 'static + Integer + Saturating + ToPrimitive + FromPrimitive + Clone + fmt::Debug,
    D: 'static + Rv<T> + Clone + fmt::Debug,
    M: 'static + Clone + fmt::Debug,
    L: 'static + Fn(&M) -> f64 + Clone + Sync,
    R: 'static + Rng
{
    fn set_adapt(&mut self, mode: AdaptationMode) {
        self.adaptor.set_mode(mode);
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
        self.step += 1;
        let current_value = self.parameter.lens.get(&model);
        let current_loglikelihood_score = loglikelihood
            .unwrap_or_else(|| (self.loglikelihood)(&model));
        let current_prior_score = self.current_prior_score
            .unwrap_or_else(|| self.parameter.prior.ln_f(&current_value));

        let current_score = current_loglikelihood_score + current_prior_score;

        // propose new value
        let scale2 = self.adaptor.get_scale().powi(2);
        let geom_p = ((4.0 * scale2 + 1.0).sqrt() - 1.0) / (2.0 * scale2);
        let proposal_dist = Geometric::new(geom_p).unwrap();
        let mag: usize = proposal_dist.draw(rng);
        let mag: T = T::from_usize(mag).unwrap();

        let proposed_new_value = if rng.gen() {
            current_value.clone().saturating_add(mag) 
        } else {
            current_value.clone().saturating_sub(mag)
        };
        let new_model = self.parameter.lens.set(&model, proposed_new_value.clone());
        let new_prior_score = self.parameter.prior.ln_f(&proposed_new_value);

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

        let update = util::metropolis_select(rng, log_alpha, proposed_new_value.clone(), current_value);
        self.adaptor.update(&update);
        let next = match update {
            util::MetroplisUpdate::Accepted(_, _) => {
                self.current_loglikelihood_score = new_loglikelihood_score;
                self.current_prior_score = Some(new_prior_score);
                ModelAndLikelihood::new(new_model, new_loglikelihood_score)
            },
            util::MetroplisUpdate::Rejected(_, _) => {
                ModelAndLikelihood::new(model, Some(current_loglikelihood_score))
            }
        };
        next
    }

    fn box_clone(&self) -> Box<SteppingAlg<M, R>> {
        Box::new(self.clone())
    }

    fn prior_draw(&self, rng: &mut R, model: M) -> M {
        self.parameter.draw(&model, rng)
    }
}



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
            x: i64,
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let parameter = Parameter::new(
            "x".to_string(),
            DiscreteUniform::new(-100, 100).unwrap(),
            make_lens!(Model, i64, x),
        );

        fn loglikelihood(m: &Model) -> f64 {
            DiscreteUniform::new(-100, 100).unwrap().ln_f(&m.x)
        }

        let alg_start = DiscreteSRWM::new(parameter, loglikelihood, Some(100.0)).unwrap();

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0 };
            let runner = Runner::new(alg_start.clone())
                .warmup(0)
                .chains(1)
                .thinning(10);

            let results: Vec<Vec<Model>> = runner.run(&mut rng, m);


            let samples: Vec<i64> = results
                .iter()
                .map(|chain| -> Vec<i64> {
                    chain.iter().map(|g| g.x).collect()
                }).flatten()
                .collect();

            let (_, p) =
                ks_test(&samples, |s| DiscreteUniform::new(-100, 100).unwrap().cdf(&s));
            p > P_VAL
        });
        assert!(passed);
    }

    #[test]
    fn uniform_posterior_warmup() {
        #[derive(Copy, Clone, Debug)]
        struct Model {
            x: i64,
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let parameter = Parameter::new(
            "x".to_string(),
            DiscreteUniform::new(-100, 100).unwrap(),
            make_lens!(Model, i64, x),
        );

        fn loglikelihood(m: &Model) -> f64 {
            DiscreteUniform::new(-100, 100).unwrap().ln_f(&m.x)
        }

        let alg_start = DiscreteSRWM::new(parameter, loglikelihood, Some(0.7)).unwrap();

        let passed = multiple_tries(N_TRIES, |_| {
            let m = Model { x: 0 };
            let results: Vec<Vec<Model>> =
                Runner::new(alg_start.clone())
                .thinning(10)
                .chains(1)
                .run(&mut rng, m);

            let samples: Vec<i64> = results
                .iter()
                .map(|chain| -> Vec<i64> {
                    chain.iter().map(|g| g.x).collect()
                }).flatten()
                .collect();

            let (_, p) =
                ks_test(&samples, |s| DiscreteUniform::new(-100, 100).unwrap().cdf(&s));
            p > P_VAL
        });
        assert!(passed);
    }
}
