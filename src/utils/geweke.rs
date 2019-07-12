//! Geweke Test for Sampler Correctness


use rand::Rng;
use crate::steppers::{SteppingAlg, AdaptationMode};
use core::borrow::BorrowMut;

/// Configuration Struct for Geweke Tests
#[derive(Clone, Copy, PartialEq)]
pub struct GewekeTestConfig {
    /// Number of draws to take from distributions
    pub sample_size: usize,
    /// Number of draws to discard between effective draws
    pub thinning: usize,
    /// Initial draws to be discarded
    pub warmup: usize,
    /// Maximum p-value for KS-Test
    pub max_p: f64,
}


/// Implements the Geweke Joint Distribution Test
/// # Geweke Joint Distribution Test
/// More info can be found [here](http://qed.econ.queensu.ca/pub/faculty/ferrall/quant/papers/04_04_29_geweke.pdf)
pub trait GewekeJDTest<M, R>
where
    M: Clone + Default,
    R: Rng,
{
    /// Statistic function
    fn g(&self, m: M) -> f64;

    /// Log-likelihood function
    fn log_likelihood(&self, model: &M) -> f64;

    /// Device to create a stepper for testing against
    fn create_stepper(&self) -> Box<dyn SteppingAlg<M, R>>;

    /// method to resample from the data generating process
    fn resample_data(&self, alg: &mut dyn SteppingAlg<M, R>, model: M, rng: &mut R) -> M;

    /// Takes MCMC samples from stepper
    fn resample_params(&self, alg: &mut dyn SteppingAlg<M, R>, m: M, rng: &mut R) -> M {
        alg.step_with_log_likelihood(rng, m, None).model
    }

    /// Maginal Conditional Simulator from Geweke Test
    fn marginal_conditional_simulator(&self, rng: &mut R, model: M, config: &GewekeTestConfig) -> Vec<f64>
    {
        let mut stepper_box = self.create_stepper();
        let stepper: &mut dyn SteppingAlg<M, R> = stepper_box.borrow_mut();
        stepper.set_adapt(AdaptationMode::Disabled);

        let prior_draw = stepper.prior_draw(rng, model.clone());


        (0..).map(|_| self.resample_data(
            stepper,
            prior_draw.clone(),
            rng
        ))
            .skip(config.warmup)
            .step_by(config.thinning)
            .take(config.sample_size)
            .map(|m| self.g(m))
            .collect()
    }

    /// Successive Conditional Simulator from Geweke test
    fn successive_conditional_simulator(&self, rng: &'static mut R, model: M, config: &GewekeTestConfig) -> Vec<f64>
    {

        let mut stepper_box = self.create_stepper();
        let stepper: &mut dyn SteppingAlg<M, R> = stepper_box.borrow_mut();
        stepper.set_adapt(AdaptationMode::Disabled);

        let prior_draw = stepper.prior_draw(rng, model.clone());
        let resampled_data = self.resample_data(stepper, prior_draw, rng);

        (0..)
            .scan(resampled_data, |m, _| {
                let next_params = self.resample_params(stepper, m.clone(), rng);
                *m = self.resample_data(
                    stepper,
                    next_params,
                    rng
                );
                Some(m.clone())
            })
            .skip(config.warmup)
            .step_by(config.thinning)
            .take(config.sample_size)
            .map(|m| self.g(m))
            .collect()
    }

    /// Run the test
    /// Boolean result returns the success condition
    fn test(&self, rng: &'static mut R, config: &GewekeTestConfig) -> bool {

        let _mcs: Vec<f64> = self.marginal_conditional_simulator(rng, M::default(), config);

        let _scs: Vec<f64> = self.successive_conditional_simulator(rng, M::default(), config);

        // let (_, p) = ks_two_sample(mcs, scs);
        // p < config.max_p
        false
    }
}
