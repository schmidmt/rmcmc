use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use rand::prelude::*;
use rayon;
use crate::steppers::SteppingAlg;

use crate::runner::utils;

pub struct Runner<M, A, R>
where
    M: 'static + Clone + Send + Sync,
    A: 'static + SteppingAlg<M, R> + Send + Sync + Clone,
    R: 'static + SeedableRng + Rng,
{
    pub stepper: A,
    pub n_chains: usize,
    pub warmup_steps: usize,
    pub samples: usize,
    pub keep_warmup: bool,
    pub thinning: usize,
    phantom_m: PhantomData<M>,
    phantom_a: PhantomData<A>,
    phantom_r: PhantomData<R>,
}

impl<M, A, R> Clone for Runner<M, A, R>
where
    M: Clone + Sync + Send,
    A: SteppingAlg<M, R> + Send + Sync + Clone,
    R: SeedableRng + Rng,
{
    fn clone(&self) -> Self {
        Runner {
            stepper: self.stepper.clone(),
            n_chains: self.n_chains,
            warmup_steps: self.warmup_steps,
            samples: self.samples,
            keep_warmup: self.keep_warmup,
            thinning: self.thinning,
            phantom_m: PhantomData,
            phantom_a: PhantomData,
            phantom_r: PhantomData,
        }
    }
}

impl<M, A, R> Runner< M, A, R>
where
    M: Clone + Sync + Send,
    A: SteppingAlg<M, R> + Send + Sync + Clone,
    M: 'static,
    A: 'static,
    R: SeedableRng + Rng + Send + Sync,
{
    pub fn new(stepper: A) -> Runner<M, A, R> {
        Runner {
            stepper,
            n_chains: 1,
            warmup_steps: 1000,
            samples: 1000,
            keep_warmup: false,
            thinning: 1,
            phantom_m: PhantomData,
            phantom_a: PhantomData,
            phantom_r: PhantomData,
        }
    }

    pub fn chains(&self, n_chains: usize) -> Self {
        Runner {
            n_chains,
            ..(*self).clone()
        }
    }

    pub fn warmup(&self, steps: usize) -> Self {
        Runner {
            warmup_steps: steps,
            ..(*self).clone()
        }
    }

    pub fn keep_warmup(&self) -> Self {
        Runner {
            keep_warmup: true,
            ..(*self).clone()
        }
    }

    pub fn drop_warmup(&self) -> Self {
        Runner {
            keep_warmup: false,
            ..(*self).clone()
        }
    }

    pub fn samples(&self, steps: usize) -> Self {
        Runner {
            samples: steps,
            ..(*self).clone()
        }
    }

    pub fn thinning(&self, thinning: usize) -> Self {
        assert!(thinning > 0, "thinning must be greater than 0.");
        Runner {
            thinning,
            ..(*self).clone()
        }
    }


    /// Run the steppers specified with this config.
    pub fn run(&self, rng: &mut R, init_model: M) -> Vec<Vec<M>>
    {
        let thinning = self.thinning;
        let keep_warmup = self.keep_warmup;
        let warmup_steps = self.warmup_steps;
        let n_chains = self.n_chains;
        let n_samples = self.samples;

        let rng = Arc::new(RwLock::new(rng));

        let results = Arc::new(RwLock::new({
            Vec::with_capacity(n_chains)
        }));

        rayon::scope(|scope| {
            (0..n_chains).for_each(|_| {
                let results = results.clone();
                let init_model = init_model.clone();
                let results = results.clone();
                let stepper = self.stepper.clone();
                let rng = Arc::clone(&rng);
                scope.spawn(move |_| {
                    let draws = utils::draw_from_stepper::<M, A, R>(rng, stepper, init_model, n_samples, warmup_steps, thinning, keep_warmup);
                    let mut res = results.write().unwrap();
                    res.push(draws);
                })
            });
        });
        let draws = results.read().unwrap().to_vec();
        draws
    }
}


#[cfg(test)]
mod tests {
    extern crate test;
    use super::Runner;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use crate::steppers::SteppingAlg;

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Model {
        i: i32
    }

    /*
    #[use_mocks]
    #[test]
    fn step_gets_called_expected_number_of_times() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut mock_stepper = new_mock!(SteppingAlg<Model, StdRng>);

        given! {
            <mock_stepper as SteppingAlg<Model, StdRng>>::{
                step(any_value(), any_value()) then_return_from |i: i32| {Model { i: i + 1 }} always;
            };
        }
        
        let runner = Runner::new(mock_stepper)
            .warmup_steps(123)
            .samples(321);
    }
    */
}
