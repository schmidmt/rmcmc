use rand::prelude::*;
use rayon;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use crate::runner::utils::draw_from_stepper;
use crate::steppers::{StepperBuilder, SteppingAlg};

/// # Runner for a stepper
///
/// # Example
/// ```rust
/// use rmcmc::steppers::Mock;
/// use rmcmc::{Runner, Parameter};
/// use rand::prelude::StdRng;
/// use rand::SeedableRng;
/// use rmcmc::steppers::srwm::SRWMBuilder;
/// use rv::dist::Gaussian;
///
/// #[derive(Clone, Copy)]
/// struct Model {
///     x: f64
/// }
///
/// let parameter = Parameter::new(
///     "x".to_owned(),
///     Gaussian::standard(),
///     make_lens!(Model, f64, x)
/// );
///
/// let builder = SRWMBuilder::new(&parameter);
///
/// let mut rng = StdRng::seed_from_u64(0u64);
///
/// let ll = |&m: Model| {
///   Gaussian::standard().ln_f(m);
/// };
///
/// let runner = Runner::new(&builder, &ll)
///     .keep_warmup()
///     .chains(2)
///     .thinning(5)
///     .samples(1000);
///
/// let sample = runner.run(&mut rng, Model { x: 0.0 });
/// ```
pub struct Runner<'a, M, A, R, L>
where
    M: Clone + Send + Sync,
    A: SteppingAlg<M> + Send + Sync + Clone,
    R: SeedableRng + Rng + Sync + Send,
    L: Fn(&M) -> f64 + Sync + Send,
{
    /// Stepper which will instrument the Markov Chain
    stepper_builder: &'a dyn StepperBuilder<M, R, L>,
    /// Number of concurrent chains to draw from
    n_chains: usize,
    /// Number of steps to adapt during
    warmup_steps: usize,
    /// Number of draws after warmup
    samples: usize,
    /// Should the warmup draws be kept
    keep_warmup: bool,
    /// Draws to consume between outer draws
    thinning: usize,
    /// Log Likelihood Function
    log_likelihood: &'a L,
    phantom_m: PhantomData<M>,
    phantom_a: PhantomData<&'a A>,
    phantom_r: PhantomData<R>,
}

impl<'a, M, A, R, L> Clone for Runner<'a, M, A, R, L>
where
    M: Clone + Sync + Send,
    A: SteppingAlg<M> + Send + Sync + Clone,
    R: SeedableRng + Rng + Sync + Send,
    L: Fn(&M) -> f64 + Sync + Send,
{
    fn clone(&self) -> Self {
        Self {
            stepper_builder: self.stepper_builder,
            n_chains: self.n_chains,
            warmup_steps: self.warmup_steps,
            samples: self.samples,
            keep_warmup: self.keep_warmup,
            thinning: self.thinning,
            log_likelihood: self.log_likelihood,
            phantom_m: PhantomData,
            phantom_a: PhantomData,
            phantom_r: PhantomData,
        }
    }
}

impl<'a, M, A, R, L> Runner<'a, M, A, R, L>
where
    M: Clone + Sync + Send,
    A: SteppingAlg<M> + Send + Sync + Clone,
    R: Rng + SeedableRng + Sync + Send,
    L: Fn(&M) -> f64 + Sync + Send,
{
    /// Create a new Runner with a given stepper
    pub fn new(
        stepper_builder: &'a (dyn StepperBuilder<M, R, L> + 'a),
        log_likelihood: &'a L,
    ) -> Self {
        Self {
            stepper_builder,
            n_chains: 1,
            warmup_steps: 1000,
            samples: 1000,
            keep_warmup: false,
            thinning: 1,
            log_likelihood,
            phantom_m: PhantomData,
            phantom_a: PhantomData,
            phantom_r: PhantomData,
        }
    }

    /// Set the number of chains to run
    pub fn chains(&self, n_chains: usize) -> Self {
        Runner {
            n_chains,
            ..(*self).clone()
        }
    }

    /// Set the number of warmup draws to consume
    pub fn warmup(&self, steps: usize) -> Self {
        Runner {
            warmup_steps: steps,
            ..(*self).clone()
        }
    }

    /// Keep warmup draws
    pub fn keep_warmup(&self) -> Self {
        Runner {
            keep_warmup: true,
            ..(*self).clone()
        }
    }

    /// Discard the warmup draws
    pub fn drop_warmup(&self) -> Self {
        Runner {
            keep_warmup: false,
            ..(*self).clone()
        }
    }

    /// Set the number of draws to return
    pub fn samples(&self, steps: usize) -> Self {
        Runner {
            samples: steps,
            ..(*self).clone()
        }
    }

    /// Set the amount of thinning (dropped intermediate draws)
    pub fn thinning(&self, thinning: usize) -> Self {
        assert!(thinning > 0, "thinning must be greater than 0.");
        Runner {
            thinning,
            ..(*self).clone()
        }
    }

    /// Run the steppers specified with this config.
    ///
    /// # Arguments
    /// `rng` - Random number generator
    /// `init_model` - Initial model to star the stepping process with.
    pub fn run(
        &self,
        rng: &'a mut R,
        init_model: M,
    ) -> crate::runner::result::Result<Vec<Vec<M>>> {
        let thinning = self.thinning;
        let keep_warmup = self.keep_warmup;
        let warmup_steps = self.warmup_steps;
        let n_chains = self.n_chains;
        let n_samples = self.samples;
        let builder = self.stepper_builder;
        let log_likelihood = self.log_likelihood;

        let rng = Arc::new(RwLock::new(rng));

        let results = Arc::new(RwLock::new({ Vec::with_capacity(n_chains) }));

        rayon::scope(|scope| {
            (0..n_chains).for_each(|_| {
                let results = results.clone();
                let init_model = init_model.clone();
                let seed: u64 =
                    rng.write().expect("Failed to access RNG").gen();

                scope.spawn(move |_| {
                    let mut rng = R::seed_from_u64(seed);
                    let stepper = builder.build(&mut rng, log_likelihood);
                    let draws = draw_from_stepper(
                        stepper,
                        init_model,
                        n_samples,
                        warmup_steps,
                        thinning,
                        keep_warmup,
                    );
                    let mut res = results.write().unwrap();
                    res.push(draws);
                })
            });
        });

        let results = results.read()?;
        Ok(results.to_vec())
    }
}

#[cfg(test)]
mod tests {
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Model {
        i: i32,
    }

    /*
    #[use_mocks]
    #[test]
    fn step_gets_called_expected_number_of_times() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut mock_stepper = new_mock!(SteppingAlg<Model, StdRng>);

        given! {
            <mock_stepper as SteppingAlg<Model, StdRng>>::{
                step(any_value(), any_value()) then_return_from |i: i32| {Model { i: i + 1 }} always;
            };
        }

        let runner = Runner::new(mock_stepper)
            .warmup_steps(123)
            .samples(321);
    }
    */
}
