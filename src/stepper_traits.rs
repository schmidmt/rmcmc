use rand::Rng;

/// Trait for Stepping Algorithms
pub trait SteppingAlg<'a, Model, RNG>: Sync + Send
    where
        Model: Clone,
{
    /// Take one step with the given stepper
    fn step(&mut self, rng: &mut RNG, model: Model) -> Model;
    /// Take a step as `step` would, but use the precomputed
    fn step_with_log_likelihood(&mut self, rng: &mut RNG, model: Model, log_likelihood: Option<f64>) -> (Model, f64);
    /// Update a model from the stepper's parameter's prior
    fn draw_prior(&self, rng: &mut RNG, m: Model) -> Model;

    /// Take multiple steps before returning model
    fn multiple_steps(&mut self, rng: &mut RNG, model: Model, steps: usize) -> Model {
        (0..steps).fold(model, |cur, _| self.step(rng, cur))
    }

    /// Return a sample from this stepper
    fn sample(&mut self, rng: &mut RNG, model: Model, size: usize, thinning: usize) -> Vec<Model> {
        (0..size).scan(model, |m, _| {
            Some(self.multiple_steps(rng, m.clone(), thinning))
        }).collect()
    }
}

/// Builder for Steppers
pub trait StepperBuilder<'a, Model, RNG>: Sync + Send
    where
        RNG: Rng,
        Model: Clone,
{
    /// Build the given Stepping Algorithm
    fn build(&self) -> Box<dyn SteppingAlg<'a, Model, RNG> + 'a>;
}