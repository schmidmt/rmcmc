use super::{AdaptationMode, AdaptationStatus};
use rand::Rng;

/// Container for an updated model and it's likelihood
#[derive(Clone, PartialEq, Default)]
pub struct ModelAndLikelihood<M> {
    /// Resulting model
    pub model: M,
    /// Optional log likelihood for resulting model
    pub log_likelihood: Option<f64>,
}

impl<M> ModelAndLikelihood<M> {
    /// Create a new ModelAndLikelihood
    /// # Arguments
    /// * `model` - Model resulting from a stepping operation
    /// * `log_likelihood` - Optional resulting log likelihood for `model`
    pub fn new(model: M, log_likelihood: Option<f64>) -> Self {
        Self {
            model,
            log_likelihood,
        }
    }
}

/// A stepping algorithm which draws the next stage from the Markov Chain.
pub trait SteppingAlg<M>: Sync + Send {
    /// Advance the parameters by one step.
    /// # Arguments
    /// * `rng` - random number generator.
    /// * `model` - model to step.
    fn step(&mut self, model: M) -> M {
        self.step_with_log_likelihood(model, None).model
    }

    /// Advance by one step but with a given log_likelihood.
    /// Used to avoid recalculating the log likelihood.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `model` - Model to step
    /// * `log_likelihood` - Optional type containing the current model's log likelihood.
    fn step_with_log_likelihood(
        &mut self,
        model: M,
        log_likelihood: Option<f64>,
    ) -> ModelAndLikelihood<M>;
    /// Set the adaptation mode
    fn set_adapt(&mut self, mode: AdaptationMode);
    /// Return the adaptation status.
    fn adapt(&self) -> AdaptationStatus;
    /// Draw from Prior
    fn prior_draw(&mut self, model: M) -> M;
}


impl<M> SteppingAlg<M> for Box<dyn SteppingAlg<M>> {
    fn step_with_log_likelihood(&mut self, model: M, log_likelihood: Option<f64>) -> ModelAndLikelihood<M> {
        self.step_with_log_likelihood(model, log_likelihood)
    }

    fn set_adapt(&mut self, mode: AdaptationMode) {
        self.set_adapt(mode);
    }

    fn adapt(&self) -> AdaptationStatus {
        self.adapt()
    }

    fn prior_draw(&mut self, model: M) -> M {
        self.prior_draw(model)
    }
}

/// Builder for a Stepper
pub trait StepperBuilder<M, R, L>: Send + Sync
where
    R: Rng + Send + Sync,
    M: Clone + Sync + Send,
    L: Fn(&M) -> f64 + Send + Sync,
{
    /// The build action for this stepper
    fn build<'b>(&self, rng: &'b mut R, log_likelihood: &'b L)
        -> Box<dyn SteppingAlg<M> + 'b>;
}
