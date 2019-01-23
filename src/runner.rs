//! Runner for a set of stepper algorithm (i.e. a markov chain)

use std::marker::PhantomData;
use std::thread;
use steppers::{SteppingAlg, AdaptationMode};
use rand::prelude::*;
use std::sync::Arc;

pub struct Runner<M, A>
where
    M: Clone + Send + Sync,
    A: SteppingAlg<M, StdRng> + Send + Sync + Clone,
{
    pub stepper: A,
    pub n_chains: usize,
    pub warmup_steps: usize,
    pub samples: usize,
    pub keep_warmup: bool,
    pub thinning: usize,
    phantom_m: PhantomData<M>,
}

impl<M, A> Clone for Runner<M, A>
where
    M: Clone + Sync + Send,
    A: 'static + SteppingAlg<M, StdRng> + Send + Sync + Clone,
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
        }
    }
}

impl<M, A> Runner<M, A>
where
    M: Clone + Sync + Send,
    A: 'static + SteppingAlg<M, StdRng> + Send + Sync + Clone,
{
    pub fn new(stepper: A) -> Runner<M, A> {
        Runner {
            stepper,
            n_chains: 1,
            warmup_steps: 1000,
            samples: 1000,
            keep_warmup: false,
            thinning: 1,
            phantom_m: PhantomData,
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
            thinning: thinning,
            ..(*self).clone()
        }
    }

    /// Run the steppers specified with this config.
    pub fn run(&'static self , init_model: M) -> Vec<Vec<M>>
    where
        M: 'static,
    {
        let seed = [0; 32];
        let mut rng: Arc<Box<StdRng>> = Arc::new(Box::new(StdRng::from_seed(seed)));

        let init = init_model;
        let thread_handles: Vec<thread::JoinHandle<_>> = (0..self.n_chains)
            .map(|i| {
                let n_samples = self.samples;
                let warmup_steps = self.warmup_steps;
                let _thinning = self.thinning;

                thread::Builder::new()
                    .name(format!("sampling thread {}", i))
                    .spawn(move || {
                        let mut stepper = self.stepper.clone();
                        let mut rng = SeedableRng::from_rng(
                            Arc::make_mut(&mut rng)
                            ).unwrap();
                        
                        // let prior_sample = stepper.prior_sample(&mut rng, init_model);
                        let prior_sample = init.clone();

                        //TODO - Randomly initialize all model values

                        // WarmUp
                        stepper.set_adapt(AdaptationMode::Enabled);

                        let mut warmup_steps = if self.keep_warmup {
                            let mp = (0..warmup_steps).fold(prior_sample, |m, _| {
                                stepper.step(&mut rng, m)
                            });
                            vec![mp]
                        } else {
                            (0..warmup_steps).scan(prior_sample, |m, _| { 
                                let mc = m.clone();
                                let mp = stepper.step(&mut rng, mc);
                                Some(mp)
                            }).collect()
                        };

                        // Draw the steps from the chain
                        stepper.set_adapt(AdaptationMode::Disabled);

                        let warmed_model = warmup_steps.last().unwrap().clone();
                        let steps: Vec<M> = (0..n_samples)
                            .scan(warmed_model, |m, _| {
                                let mc = m.clone();
                                let mp = stepper.step(&mut rng, mc);
                                Some(mp)
                            }).collect();

                        if self.keep_warmup {
                            warmup_steps.extend(steps);
                            warmup_steps
                        } else {
                            steps
                        }
                    })
                    .unwrap()
            }).collect();

        // get results from threads
        let mut results = vec![];
        for child in thread_handles {
            results.push(child.join().unwrap());
        }
        results
    }
}
