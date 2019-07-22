use crate::steppers::adaptors::AdaptState;
use crate::SteppingAlg;
use rand::Rng;

/// Group wrapper for multiple steppers
pub struct Group<'a, Model, RNG>
where
    RNG: Rng,
{
    sub_steppers: Vec<Box<(dyn SteppingAlg<'a, Model, RNG> + 'a)>>,
    current_log_likelihood: Option<f64>,
}

impl<'a, Model, RNG> Group<'a, Model, RNG>
where
    RNG: Rng,
{
    /// Create a group of steppers to act in unison
    /// __Should__ be constructed with `GroupBuilder`
    ///
    /// # Parameters
    /// * `sub_steppers` - Steppers which compose the group
    ///
    /// # Example
    ///```rust
    /// use rmcmc::SteppingAlg;
    /// use rmcmc::StepperBuilder;
    /// use rmcmc::steppers::srwm::SRWMBuilder;
    /// use rmcmc::Parameter;
    /// use rmcmc::{Lens, make_lens};
    /// use rv::dist::Gaussian;
    /// use rmcmc::steppers::group::Group;
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    ///
    /// #[derive(Clone)]
    /// struct Model {
    ///     x: f64,
    ///     y: f64
    /// }
    ///
    /// let log_likelihood = |m: &Model| {
    ///     - (m.x * m.x + m.y * m.y)
    /// };
    ///
    /// let x_param = Parameter::new(
    ///     Gaussian::standard(),
    ///     make_lens!(Model, f64, x)
    /// );
    ///
    /// let srwm_1 = SRWMBuilder::new(
    ///     &x_param,
    ///     &log_likelihood,
    ///     0.0,
    ///     1.0
    /// ).build();
    ///
    /// let y_param = Parameter::new(
    ///     Gaussian::standard(),
    ///     make_lens!(Model, f64, y)
    /// );
    ///
    /// let srwm_2 = SRWMBuilder::new(
    ///     &y_param,
    ///     &log_likelihood,
    ///     0.0,
    ///     1.0
    /// ).build();
    ///
    /// let mut rng = StdRng::seed_from_u64(0xABC);
    /// let mut stepper = Group::new(vec![srwm_1, srwm_2]);
    /// let sample = stepper.sample(&mut rng, Model { x: 0.0, y: 0.0 }, 10, 1);
    ///```
    pub fn new(
        sub_steppers: Vec<Box<dyn SteppingAlg<'a, Model, RNG> + 'a>>,
    ) -> Self {
        Self {
            sub_steppers,
            current_log_likelihood: None,
        }
    }
}

impl<'a, Model, RNG> SteppingAlg<'a, Model, RNG> for Group<'a, Model, RNG>
where
    Model: Clone + Send + Sync,
    RNG: Rng + Send + Sync,
{
    fn step(&mut self, rng: &mut RNG, model: Model) -> Model {
        let current_ll = self.current_log_likelihood;
        self.step_with_log_likelihood(rng, model, current_ll).0
    }

    fn step_with_log_likelihood(
        &mut self,
        rng: &mut RNG,
        model: Model,
        log_likelihood: Option<f64>,
    ) -> (Model, f64) {
        let (next_model, next_ll) = self.sub_steppers.iter_mut().fold(
            (model, log_likelihood),
            |(m, ll), s| {
                let (nm, nll) = s.step_with_log_likelihood(rng, m, ll);
                (nm, Some(nll))
            },
        );
        self.current_log_likelihood = next_ll;
        (next_model, next_ll.unwrap())
    }

    fn draw_prior(&self, rng: &mut RNG, m: Model) -> Model {
        self.sub_steppers
            .iter()
            .fold(m, |model, stepper| stepper.draw_prior(rng, model))
    }

    fn adapt_enable(&mut self) {
        self.sub_steppers
            .iter_mut()
            .for_each(|s| (*s).adapt_enable());
    }

    fn adapt_disable(&mut self) {
        self.sub_steppers
            .iter_mut()
            .for_each(|s| (*s).adapt_disable());
    }

    fn adapt_state(&self) -> AdaptState {
        self.sub_steppers
            .iter()
            .fold(AdaptState::NotApplicable, |state, s| {
                state.merge(s.adapt_state())
            })
    }
}
