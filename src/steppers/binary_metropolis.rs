//! # Binary Metropolis
//! Preforms sampling on a single binary variable.

use std::fmt;
extern crate rand;
use rand::Rng;

use rv::traits::Rv;
use parameter::Parameter;

use steppers::{SteppingAlg, AdaptationStatus, AdaptationMode, util};
use statistics::Statistic;
use steppers::adaptor::{ScaleAdaptor, SimpleAdaptor};


#[derive(Clone)]
pub struct BinaryMetropolis<D, T, M, L>
where
    T: Clone,
    D: Rv<T> + Clone + fmt::Debug,
    M: 'static + Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync
{
    pub parameter: Parameter<D, T, M>,
    pub log_likelihood: L,
    pub current_score: Option<f64>,
    adaptor: SimpleAdaptor<T>,
}

impl<D, T, M, L> std::fmt::Debug for BinaryMetropolis<D, T, M, L>
where
    T: Clone,
    D: Rv<T> + Clone + fmt::Debug,
    M: 'static + Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BinaryMetropolis {{ parameter: ")?;
        self.parameter.fmt(f)?;
        write!(f, " }}")
    }
}
    


impl<D, T, M, L> BinaryMetropolis<D, T, M, L> 
where
    D: Rv<T> + Clone + fmt::Debug,
    T: Clone + fmt::Debug,
    M: 'static + Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
{
    pub fn new(
        parameter: Parameter<D, T, M>,
        log_likelihood: L
    ) -> Option<Self> {
        let adaptor = SimpleAdaptor::new(0.5, 50);
        Some(Self {
            parameter,
            log_likelihood,
            current_score: None,
            adaptor
        })
    }
}


impl<D, L, M, R> SteppingAlg<M, R> for BinaryMetropolis<D, Vec<bool>, M, L>
where
    D: Rv<Vec<bool>> + Clone + fmt::Debug,
    M: 'static + Clone + fmt::Debug,
    L: Fn(&M) -> f64 + Clone + Sync,
    R: Rng,
{
    fn set_adapt(&mut self, _mode: AdaptationMode) {}
    fn get_adapt(&self) -> AdaptationStatus {
        self.adaptor.get_mode()
    }
    fn get_statistics(&self) -> Vec<Statistic<M, R>> {
        Vec::new()
    }
    fn reset(&mut self) {}

    fn step(&mut self, rng: &mut R, model: M) -> M {
        let p = 1.0 - (0.5f64).powf(self.adaptor.get_scale());
        let mut m = model.clone();
        let mut log_p = match self.current_score {
            Some(p) => p,
            None => (self.log_likelihood)(&model)
        };
        let mut value = self.parameter.lens.get(&model);
        (0..value.len()).for_each(|idx| {
            if rng.gen::<f64>() < p {
                let mut proposed_value = value.clone();
                proposed_value[idx] = !proposed_value[idx];
                self.parameter.lens.set_in_place(&mut m, proposed_value.clone());
                let proposed_log_p = (self.log_likelihood)(&m);
                
                let update = util::metropolis_select(rng, proposed_log_p - log_p, proposed_value.clone(), value.clone());
                self.adaptor.update(&update);
                match update {
                    util::MetroplisUpdate::Accepted(_, _) => {
                        value[idx] = proposed_value[idx];
                        log_p = proposed_log_p;
                    },
                    util::MetroplisUpdate::Rejected(_, _) => {}
                }
            }
        });

        self.current_score = Some(log_p);
        self.parameter.lens.set_in_place(&mut m, value);
        m
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
    use utils::MultiRv;

    const P_VAL: f64 = 0.2;
    const N_TRIES: usize = 10;
    const SEED: [u8; 32] = [0; 32];

    #[test]
    fn binary_gaussian_mixture() {
        let mut rng = rand::rngs::StdRng::from_seed(SEED);

        #[derive(Clone, Debug)]
        struct Model {
            p: Vec<bool>,
        }

        let dims = 30;

        let dist = MultiRv::new(dims, Bernoulli::new(0.5).unwrap());
        let lens = Lens::new(
            |s: &Model| s.p.clone(),
            |s: &Model, x: Vec<bool>| Model { p: x, ..*s }
        );

        let parameter = Parameter::new(
            "p".to_string(),
            dist.clone(),
            make_lens_clone!(Model, Vec<bool>, p)
        );
    
        let passed = multiple_tries(N_TRIES, |_| {
            let parameter = parameter.clone();
            let p: f64 = 0.75;
            // Generate samples from gmm with two gaussians.
            let g1 = Gaussian::new(0.0, 1.0).unwrap();
            let g2 = Gaussian::new(3.0, 0.9).unwrap();
            let samples: Vec<f64> = (0..dims).map(|_| {
                if rng.gen::<f64>() < p {
                   g1.draw(&mut rng)
                } else {
                   g2.draw(&mut rng)
                }
            }).collect();

            // Log Likelihood Calculation
            let log_likelihood = move |m: &Model| {
                m.p.iter().zip(samples.iter()).map(|(&a, y)| {
                    if a {
                        g1.ln_f(y)
                    } else {
                        g2.ln_f(y)
                    }
                }).sum()
            };

            // Create algorithm and run sampler
            let alg = BinaryMetropolis::new(parameter, log_likelihood).unwrap();
            let m = Model { p: dist.draw(&mut rng) };
            
            let runner = Runner::new(alg)
                .thinning(1)
                .chains(1)
                .run(&mut rng, m);

            let draws: Vec<Vec<bool>> = runner
                .iter()
                .flat_map::<Vec<Vec<bool>>, _>(|c| c.iter().map(|g| g.p.clone()).collect())
                .collect();

            let mut inferred_p: Vec<f64> = draws.iter().map(|x| {
                let true_count = x.iter().fold(0, |a, &y| {
                    if y {
                        a + 1
                    } else {
                        a
                    }
                });
                (true_count as f64) / (dims as f64)
            }).collect();

            inferred_p.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let (iqr_l, iqr_u) = (inferred_p[250], inferred_p[750]);
            p > iqr_l && p < iqr_u
        });
        assert!(passed);
    }
}
