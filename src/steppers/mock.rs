
use std::fmt;
use steppers::{SteppingAlg, AdaptationMode, AdaptationStatus};
use statistics::Statistic;

#[derive(Clone)]
pub struct Mock<M, F> 
where
    M: Clone,
    F: Fn(M) -> M
{
    model: M,
    update: F,
}

impl<M, F> Mock< M, F> 
where
    M: Clone,
    F: Fn(M) -> M
{
    pub fn new(model: M, update: F) -> Mock<M, F> {
        Mock {
            model,
            update
        }
    }
}

impl<M, F> fmt::Debug for Mock<M, F>
where
    M: Clone,
    F: Fn(M) -> M
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mock {{ }}")
    }
}


impl<M, F, R> SteppingAlg<M, R> for Mock<M, F>
where
    M: Clone,
    R: rand::RngCore,
    F: Fn(M) -> M
{
    fn step(&mut self, _rng: &mut R, model: M) -> M {
        (self.update)(model)
    }

    fn set_adapt(&mut self, _mode: AdaptationMode) {}
    fn get_adapt(&self) -> AdaptationStatus {
        AdaptationStatus::Disabled
    }

    fn get_statistics(&self) -> Vec<Statistic<M, R>> {
        Vec::new()
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;
    use runner::Runner;
    use rand::SeedableRng;
    const SEED: [u8; 32] = [0; 32];

    #[test]
    fn update_can_do_simple_updates() {
        let update = |x: i32| x + 1;
        let init: i32 = 1;
        let mut mock = Mock::new(init, update);
        let mut rng = rand::rngs::StdRng::from_seed(SEED);
        let a = mock.step(&mut rng, init);
        assert_eq!(a, 2);
        let b = mock.step(&mut rng, a);
        assert_eq!(b, 3);
    }
}
