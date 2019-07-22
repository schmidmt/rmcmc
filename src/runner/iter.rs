use rand::Rng;
use crate::{SteppingAlg, StepperBuilder, InitializationMode};

pub struct RunnerIter<'a, M, RNG>
    where
        M: Clone,
        RNG: Rng,
{
    draws: usize,
    warm_up: usize,
    thinning: usize,
    chains: usize,
    builder: &'a dyn StepperBuilder<'a, Model, RNG>,
    init: InitializationMode<Model>,
    keep_warm_up: bool,
    stepper: Box<dyn SteppingAlg<'a, M, RNG> + 'a>,
}

impl<'a, M, RNG> Iterator for RunnerIter<'a, M, R> {
    type Item = M;

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}
