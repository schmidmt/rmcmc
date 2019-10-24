// mod iter;
// pub use iter::*;

use crate::StepperBuilder;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Initialization mode for steppers' models
#[derive(Clone)]
pub enum InitializationMode<M>
    where
        M: Clone,
{
    /// Draw from the priors defined within the steppers' parameters
    DrawFromPrior,
    /// Set the given model as the initial value
    Provided(M),
}

/// Runner for drawing a sample from a posterior
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
        RNG: Rng + SeedableRng,
{
    /// Create a new Runner
    ///
    /// # Parameters
    /// * `builder` - Builder which creates a given stepper
    ///
    /// # Example
    /// ```rust
    /// use rmcmc::{Runner, Parameter, Lens, make_lens};
    /// use rmcmc::steppers::srwm::{SRWM, SRWMBuilder};
    /// use rv::dist::{Gaussian, Poisson, Exponential, Gamma};
    /// use rmcmc::utils::log_likelihood_from_data;
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use rv::misc::ks_test;
    /// use rv::traits::Cdf;
    ///
    /// #[derive(Clone)]
    /// struct Model {
    ///     mean: f64
    /// }
    ///
    /// impl Default for Model {
    ///     fn default() -> Self {
    ///         Model { mean: 0.0 }
    ///     }
    /// }
    ///
    /// let data: Vec<u32> = vec![2, 3, 2, 1, 1, 3, 2, 4, 3, 4];
    ///
    /// let log_likelihood = log_likelihood_from_data(&data, |m: &Model| {Poisson::new(m.mean) });
    ///
    /// let parameter = Parameter::new(
    ///     Gamma::new(3.0, 3.0).unwrap(),
    ///     make_lens!(Model, f64, mean)
    /// );
    ///
    /// let builder = SRWMBuilder::new(
    ///     &parameter,
    ///     &log_likelihood,
    ///     10.0,
    ///     1.0
    /// );
    ///
    /// let runner = Runner::new(&builder)
    ///     .chains(1)
    ///     .warmup(1000)
    ///     .draws(1000);
    ///
    /// let mut rng = StdRng::seed_from_u64(0xFEED);
    /// let sample: Vec<Vec<Model>> = runner.run(&mut rng);
    ///
    /// assert_eq!(sample.get(0).unwrap().len(), 1000);
    /// ```
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

    /// Set the size of the sample to draw from each chain.
    pub fn draws(&self, samples: usize) -> Self {
        Self {
            draws: samples,
            ..(*self).clone()
        }
    }

    /// Set the number of warmup (adapting) steps to take before drawing samples steps.
    pub fn warmup(&self, warmup: usize) -> Self {
        Self {
            warm_up: warmup,
            ..(*self).clone()
        }
    }

    /// Number of chains to draw from simultaneously.
    pub fn chains(&self, chains: usize) -> Self {
        assert!(chains >= 1, "The number of chains must be one or more.");
        Self {
            chains,
            ..(*self).clone()
        }
    }

    /// Number of steps between sample draws.
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

    /// Set initial model
    pub fn initial_model(&self, model: Model) -> Self {
        Self {
            init: InitializationMode::Provided(model),
            ..(*self).clone()
        }
    }

    /// Run the given
    pub fn run(&self, rng: &mut RNG) -> Vec<Vec<Model>> {
        let seeds: Vec<u64> = (0..self.chains)
            .map(|_| {
                let seed: u64 = rng.gen();
                seed
            })
            .collect();

        seeds
            .par_iter()
            .map(|seed| {
                let mut rng = RNG::seed_from_u64(*seed);
                let mut stepper = self.builder.build();

                let init_model = match &self.init {
                    InitializationMode::DrawFromPrior => {
                        stepper.draw_prior(&mut rng, Model::default())
                    }
                    InitializationMode::Provided(m) => m.clone(),
                };

                // Warm Up
                stepper.adapt_enable();
                let mut warmup =
                    stepper.sample(&mut rng, init_model, self.warm_up, 1);
                stepper.adapt_disable();

                // Sample Generation
                let sample = stepper.sample(
                    &mut rng,
                    warmup.last().unwrap().clone(),
                    self.draws,
                    self.thinning,
                );

                if self.keep_warm_up {
                    warmup.extend(sample);
                    warmup
                } else {
                    sample
                }
            })
            .collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::stepper_mocks::MockBuilder;
    use rand::rngs::StdRng;

    #[derive(Clone)]
    struct Model {
        x: f64,
    }

    impl Default for Model {
        fn default() -> Self {
            Self { x: 0.0 }
        }
    }

    #[test]
    fn abc() {
        let builder = MockBuilder::new();
        let _runner: Runner<Model, StdRng> = Runner::new(&builder)
            .chains(1)
            .initial_model(Model { x: 0.0 });
    }
}
