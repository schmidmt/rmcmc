use crate::StepperBuilder;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Clone)]
pub enum InitializationMode<M>
where
    M: Clone
{
    DrawFromPrior,
    Provided(M),
}

pub struct Runner<'a, Model, RNG>
    where
        Model: Clone + Send + Sync + Default,
{
    draws: usize,
    warm_up: usize,
    thinning: usize,
    chains: usize,
    builder: &'a dyn StepperBuilder<'a, Model, RNG>,
    init: InitializationMode<Model>,
    keep_warm_up: bool,
}

impl<'a, Model, RNG> Clone for Runner<'a, Model, RNG>
    where
        Model: Clone + Send + Sync + Default,
{
    fn clone(&self) -> Self {
        Self {
            draws: self.draws,
            warm_up: self.warm_up,
            thinning: self.thinning,
            chains: self.chains,
            builder: self.builder,
            init: self.init.clone(),
            keep_warm_up: self.keep_warm_up,
        }
    }
}


impl<'a, Model, RNG> Runner<'a, Model, RNG>
    where
        Model: Clone + Send + Sync + Default,
        RNG: Rng + SeedableRng
{
    pub fn new(builder: &'a dyn StepperBuilder<'a, Model, RNG>) -> Self {
        Self {
            builder,
            draws: 2000,
            warm_up: 1000,
            thinning: 1,
            chains: 1,
            init: InitializationMode::DrawFromPrior,
            keep_warm_up: false,
        }
    }

    pub fn draws(&self, samples: usize) -> Self {
        Self {
            draws: samples,
            ..(*self).clone()
        }
    }

    pub fn warmup(&self, warmup: usize) -> Self {
        Self {
            warm_up: warmup,
            ..(*self).clone()
        }
    }

    pub fn chains(&self, chains: usize) -> Self {
        assert!(chains >= 1, "The number of chains must be one or more.");
        Self {
            chains,
            ..(*self).clone()
        }
    }

    pub fn thinning(&self, thinning: usize) -> Self {
        assert_ne!(thinning, 0, "Thinning cannot be lower than one.");
        Self {
            thinning,
            ..(*self).clone()
        }
    }

    /// Include warm-up in sample
    pub fn keep_warm_up(&self) -> Self {
        Self {
            keep_warm_up: true,
            ..(*self).clone()
        }
    }

    /// Do not include warm-up in sample
    pub fn discard_warm_up(&self) -> Self {
        Self {
            keep_warm_up: false,
            ..(*self).clone()
        }
    }

    // Set initial model
    pub fn initial_model(&self, model: Model) -> Self {
        Self {
            init: InitializationMode::Provided(model),
            ..(*self).clone()
        }
    }

    pub fn run(&self, rng: &mut RNG) -> Vec<Vec<Model>> {

        let seeds: Vec<u64> = (0..self.chains)
            .map(|_| {
                let seed: u64 = rng.gen();
                seed
            }).collect();

        seeds
            .par_iter()
            .map(|seed|{
                let mut rng = RNG::seed_from_u64(*seed);
                let mut stepper = self.builder.build();

                let init_model = match &self.init {
                    InitializationMode::DrawFromPrior => stepper.draw_prior(&mut rng, Model::default()),
                    InitializationMode::Provided(m) => m.clone(),
                };

                // Warm Up
                stepper.adapt_enable();
                let mut warmup = stepper.sample(&mut rng, init_model, self.warm_up, 1);
                stepper.adapt_disable();
                // Sample Generation
                let sample = stepper.sample(&mut rng, warmup.last().unwrap().clone(), self.draws, self.thinning);

                if self.keep_warm_up {
                    warmup.extend(sample);
                    warmup
                } else {
                    sample
                }
            }).collect()
    }
}


