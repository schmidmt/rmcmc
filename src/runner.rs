use crate::StepperBuilder;
use rand::Rng;

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
    warmup: usize,
    thinning: usize,
    chains: usize,
    builder: &'a dyn StepperBuilder<'a, Model, RNG>,
    init: InitializationMode<Model>
}

impl<'a, Model, RNG> Clone for Runner<'a, Model, RNG>
    where
        Model: Clone + Send + Sync + Default,
{
    fn clone(&self) -> Self {
        Self {
            draws: self.draws,
            warmup: self.warmup,
            thinning: self.thinning,
            chains: self.chains,
            builder: self.builder,
            init: self.init.clone(),
        }
    }
}


impl<'a, Model, RNG> Runner<'a, Model, RNG>
    where
        Model: Clone + Send + Sync + Default,
        RNG: Rng
{
    pub fn new(builder: &'a dyn StepperBuilder<'a, Model, RNG>) -> Self {
        Self {
            builder,
            draws: 2000,
            warmup: 1000,
            thinning: 1,
            chains: 1,
            init: InitializationMode::DrawFromPrior,
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
            warmup,
            ..(*self).clone()
        }
    }

    pub fn chains(&self, chains: usize) -> Self {
        Self {
            chains,
            ..(*self).clone()
        }
    }

    pub fn thinning(&self, thinning: usize) -> Self {
        Self {
            thinning,
            ..(*self).clone()
        }
    }

    pub fn run(&self, rng: &mut RNG) -> Vec<Vec<Model>> {
        (0..self.chains).map(|_|{
            let mut stepper = self.builder.build();

            let init_model = match &self.init {
                InitializationMode::DrawFromPrior => stepper.draw_prior(rng, Model::default()),
                InitializationMode::Provided(m) => m.clone(),
            };

            stepper.sample(rng, init_model, self.draws, self.thinning)
        }).collect()
    }
}


