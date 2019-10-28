use crate::steppers::group::Group;
use crate::{StepperBuilder, SteppingAlg};
use rand::Rng;

/// Builder for a Group Stepper
pub struct GroupBuilder<'a, Model, RNG>
where
    RNG: Rng,
    Model: Clone,
{
    sub_stepper_builders:
        Vec<&'a dyn StepperBuilder<'a, Model, RNG>>,
}

impl<'a, Model, RNG> GroupBuilder<'a, Model, RNG>
where
    RNG: Rng + Send + Sync,
    Model: Clone + Send + Sync,
{
    /// Create a new stepper group of steppers
    ///
    /// # Parameters
    /// * `sub_stepper_builders` - Vec of builders for substeppers.
    ///
    /// # Example
    /// ```rust
    /// use rmcmc::steppers::group::GroupBuilder;
    /// use rmcmc::steppers::srwm::SRWMBuilder;
    /// use rmcmc::{Parameter, Runner};
    /// use rmcmc::{make_lens, Lens};
    /// use rv::dist::{Gaussian, MvGaussian, Gamma};
    /// use rv::traits::Rv;
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use rmcmc::utils::log_likelihood_from_data;
    ///
    /// #[derive(Clone)]
    /// struct Model {
    ///     mu: f64,
    ///     s2: f64,
    /// }
    ///
    /// impl Default for Model {fn default() -> Self {
    ///         Model { mu: 0.0, s2: 1.0 }
    ///     }
    /// }
    ///
    /// let mut rng: StdRng = StdRng::seed_from_u64(0xFEED);
    ///
    /// let data: Vec<f64> = {
    ///     let g = Gaussian::new(5.0, 3.0).unwrap();
    ///     g.sample(50, &mut rng)
    /// };
    ///
    /// let log_likelihood = log_likelihood_from_data(&data, |m: &Model| {
    ///     Gaussian::new(m.mu, m.s2.sqrt())
    /// });
    ///
    /// let mu = Parameter::new_independent(
    ///     Gaussian::standard(),
    ///     make_lens!(Model, f64, mu)
    /// );
    ///
    /// let mu_stepper = SRWMBuilder::new(&mu, &log_likelihood, 0.0, 1.0);
    ///
    /// let s2 = Parameter::new_independent(
    ///     Gamma::new(3.0, 3.0).unwrap(),
    ///     make_lens!(Model, f64, s2)
    /// );
    ///
    /// let s2_stepper = SRWMBuilder::new(&s2, &log_likelihood, 1.0, 1.0);
    ///
    /// let group_builder = GroupBuilder::new(vec![&mu_stepper, &s2_stepper]);
    ///
    /// let runner = Runner::new(&group_builder)
    ///     .chains(1)
    ///     .draws(1000)
    ///     .thinning(10)
    /// ;
    ///
    /// let sample = runner.run(&mut rng);
    ///
    /// assert_eq!(sample.get(0).unwrap().len(), 1000);
    ///
    /// ```
    pub fn new(
        sub_stepper_builders: Vec<
            &'a dyn StepperBuilder<'a, Model, RNG>,
        >,
    ) -> Self {
        Self {
            sub_stepper_builders,
        }
    }
}

impl<'a, Model, RNG> StepperBuilder<'a, Model, RNG>
    for GroupBuilder<'a, Model, RNG>
where
    RNG: Rng + Send + Sync,
    Model: Clone + Send + Sync,
{
    fn build(&self) -> Box<dyn SteppingAlg<'a, Model, RNG> + 'a> {
        Box::new(Group::new(
            self.sub_stepper_builders
                .iter()
                .map(|b| b.build())
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn builder_constructs_sub_steppers() {}
}
