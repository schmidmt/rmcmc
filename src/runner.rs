//! Runner for a set of stepper algorithm (i.e. a markov chain)

extern crate rand;
use std::marker::PhantomData;
use std::thread;
use traits::*;

pub struct Runner<M, A>
where
    M: Clone + Send + Sync,
    A: SteppingAlg<M> + Send + Sync + Clone,
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
    A: 'static + SteppingAlg<M> + Send + Sync + Clone,
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
    A: 'static + SteppingAlg<M> + Send + Sync + Clone,
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
    pub fn run(&self, init_model: M) -> Vec<Vec<(M, A)>>
    where
        M: 'static,
    {
        let thread_handles: Vec<thread::JoinHandle<_>> = (0..self.n_chains)
            .map(|i| {
                let n_samples = self.samples;
                let warmup_steps = self.warmup_steps;
                let thinning = self.thinning;
                let stepper = self.stepper.clone();
                let mut model = init_model.clone();

                thread::Builder::new()
                    .name(format!("sampling thread {}", i))
                    .spawn(move || {
                        let mut rng = rand::thread_rng();
                        //TODO - Randomly initialize all model values

                        // WarmUp
                        let adapted_stepper: A = stepper.adapt_on();

                        let warmed_stepper = (0..warmup_steps)
                            .fold(adapted_stepper, |acc, _| {
                                acc.step(&mut rng, &mut model)
                            });

                        // Draw the steps from the chain
                        let stable_stepper: A = warmed_stepper.adapt_off();

                        let steps: Vec<(M, A)> = (0..n_samples)
                            .scan(
                                (model, stable_stepper),
                                |(cur_model, cur_stepper), _| {
                                    let mut this_model = cur_model.clone();
                                    *cur_stepper = (0..thinning).fold(
                                        cur_stepper.clone(),
                                        |cur, _| {
                                            cur.step(&mut rng, &mut this_model)
                                        },
                                    );
                                    *cur_model = this_model;
                                    Some((
                                        (*cur_model).clone(),
                                        (*cur_stepper).clone(),
                                    ))
                                },
                            ).collect();

                        steps
                    }).unwrap()
            }).collect();

        // get results from threads
        let mut results = vec![];
        for child in thread_handles {
            results.push(child.join().unwrap());
        }
        results
    }
}
