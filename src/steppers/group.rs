extern crate rand;
use rand::Rng;
use std::marker::PhantomData;
use traits::*;

/// Stepper Group
#[derive(Clone, Debug)]
pub struct Group<M, A>
where
    M: Clone,
    A: SteppingAlg<M> + Clone,
{
    pub steppers: Vec<A>,
    phantom_m: PhantomData<M>,
}

impl<M, A> Group<M, A>
where
    M: Clone,
    A: SteppingAlg<M> + Clone,
{
    pub fn new(steppers: Vec<A>) -> Self {
        Group {
            steppers: steppers,
            phantom_m: PhantomData,
        }
    }
}

impl<M, A> SteppingAlg<M> for Group<M, A>
where
    M: Clone,
    A: SteppingAlg<M> + Clone,
{
    fn step<R: Rng>(&self, rng: &mut R, model: &mut M) -> Self {
        let new_steppers = self
            .steppers
            .iter()
            .map(|stepper| stepper.step(rng, model))
            .collect();
        Group {
            steppers: new_steppers,
            ..*self
        }
    }

    fn adapt_on(&self) -> Self {
        Group {
            steppers: self.steppers.iter().map(|s| s.adapt_on()).collect(),
            ..*self
        }
    }

    fn adapt_off(&self) -> Self {
        Group {
            steppers: self.steppers.iter().map(|s| s.adapt_off()).collect(),
            ..*self
        }
    }
}
