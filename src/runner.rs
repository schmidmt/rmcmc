//! Runner for a set of stepper algorithm (i.e. a markov chain)

use std::marker::PhantomData;
use steppers::SteppingAlg;
use rand::prelude::*;
use rayon;
use std::sync::{Arc, RwLock};
use std::fmt;

mod utils {
    use steppers::{SteppingAlg, AdaptationMode};
    use rand::prelude::*;
    use std::sync::{Arc, RwLock};
    use std::ops::DerefMut;

    pub fn draw_from_stepper<M, A, R>(
        rng: Arc<RwLock<&mut R>>,
        stepper: A,
        init: M,
        n_draws: usize,
        n_warmup: usize,
        thinning: usize,
        keep_warmup: bool,
    ) -> Vec<M> 
    where
        M: Clone + Sync + Send,
        A: SteppingAlg<M, R> + Send + Sync + Clone,
        R: SeedableRng + Rng + std::fmt::Debug,
    {
        let mut rng = SeedableRng::from_rng(
            rng.write()
            .expect("Failed to get write access to rng")
            .deref_mut()
        ).expect("Failed to create seedable rng from input rng.");


        let mut stepper = stepper.clone();
        // let prior_sample = stepper.prior_sample(&mut rng, init_model);
        let prior_sample = init;

        //TODO - Randomly initialize all model values

        // WarmUp
        stepper.set_adapt(AdaptationMode::Enabled);

        let mut warmup_draws = if keep_warmup {
            let mp = (0..n_warmup).fold(prior_sample.clone(), |m, _| {
                stepper.step(&mut rng, m)
            });
            vec![mp]
        } else {
            (0..n_warmup).scan(prior_sample.clone(), |m, _| { 
                let mc = m.clone();
                let mp = stepper.step(&mut rng, mc);
                Some(mp)
            }).collect()
        };

        // Draw the steps from the chain
        stepper.set_adapt(AdaptationMode::Disabled);

        let warmed_model: M = if warmup_draws.is_empty() {
            prior_sample
        } else {
            warmup_draws.last().unwrap().clone()
        };
        let draws: Vec<M> = (0..(n_draws * thinning))
            .scan(warmed_model, |m, _| {
                let mc = m.clone();
                let mp = stepper.step(&mut rng, mc);
                Some(mp)
            })
            .step_by(thinning)
            .collect();
        
        if keep_warmup {
            warmup_draws.extend(draws);
            warmup_draws
        } else {
            draws
        }
    }

}


pub struct Runner<M, A, R>
where
    M: Clone + Send + Sync,
    A: SteppingAlg<M, R> + Send + Sync + Clone,
    R: SeedableRng + Rng,
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

impl<M, A, R> Runner<M, A, R>
where
    M: Clone + Sync + Send,
    A: SteppingAlg<M, R> + Send + Sync + Clone,
    M: 'static,
    A: 'static,
    R: SeedableRng + Rng + fmt::Debug + Send + Sync,
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
            thinning: thinning,
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
