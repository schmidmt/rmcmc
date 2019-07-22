use crate::steppers::adaptors::AdaptState;
use crate::{StepperBuilder, SteppingAlg};
use pseudo::Mock;
use rand::Rng;

struct MockStepper<M>
where
    M: Clone,
{
    pub step_fn: Mock<M, M>,
    pub step_with_ll_fn: Mock<(M, Option<f64>), (M, f64)>,
    pub draw_prior_fn: Mock<M, M>,
    pub adapt_change_fn: Mock<bool, ()>,
    pub adapt_status_fn: Mock<(), AdaptState>,
}

impl<'a, M> MockStepper<M>
where
    M: Clone + Default,
{
    pub fn new() -> Self {
        Self {
            step_fn: Mock::default(),
            step_with_ll_fn: Mock::default(),
            draw_prior_fn: Mock::default(),
            adapt_status_fn: Mock::default(),
            adapt_change_fn: Mock::default(),
        }
    }
}

impl<'a, Model, RNG> SteppingAlg<'a, Model, RNG> for MockStepper<Model>
where
    RNG: Rng + Send + Sync,
    Model: Clone + Send + Sync + Default,
{
    fn step(&mut self, _: &mut RNG, model: Model) -> Model {
        self.step_fn.call(model)
    }

    fn step_with_log_likelihood(
        &mut self,
        _: &mut RNG,
        model: Model,
        log_likelihood: Option<f64>,
    ) -> (Model, f64) {
        self.step_with_ll_fn.call((model, log_likelihood))
    }

    fn draw_prior(&self, _: &mut RNG, m: Model) -> Model {
        self.draw_prior_fn.call(m)
    }

    fn adapt_enable(&mut self) {
        self.adapt_change_fn.call(true)
    }

    fn adapt_disable(&mut self) {
        self.adapt_change_fn.call(false)
    }

    fn adapt_state(&self) -> AdaptState {
        self.adapt_status_fn.call(())
    }
}

pub struct MockBuilder {}

impl MockBuilder {
    pub fn new() -> Self {
        Self {}
    }
}

impl<'a, Model, RNG> StepperBuilder<'a, Model, RNG> for MockBuilder
where
    RNG: Rng + Send + Sync,
    Model: Clone + Default + Send + Sync + 'a,
{
    fn build(&self) -> Box<dyn SteppingAlg<'a, Model, RNG> + 'a> {
        Box::new(MockStepper::new())
    }
}
