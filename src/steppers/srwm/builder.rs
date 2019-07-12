use crate::steppers::adaptor::GlobalAdaptor;
use crate::steppers::{StepperBuilder, SteppingAlg};
use crate::steppers::SRWM;
use crate::Parameter;
use num::traits::{Float, One, Zero};
use rand::Rng;
use rv::traits::{Mean, Rv, Variance};

pub struct SRWMBuilder<'a, D, T, M, V>
where
    D: Rv<T> + Clone,
    T: Clone,
    V: Clone,
{
    adaptor: GlobalAdaptor<T, V>,
    parameter: &'a Parameter<D, T, M>,
}

impl<'a, D, T, M, V> Clone for SRWMBuilder<'a, D, T, M, V>
where
    D: Rv<T> + Clone,
    T: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            adaptor: self.adaptor.clone(),
            parameter: self.parameter,
        }
    }
}

impl<'a, D, T, M, V> SRWMBuilder<'a, D, T, M, V>
where
    D: Rv<T> + Clone,
    T: Zero + Clone,
    V: One + Clone,
{
    pub fn new(parameter: &'a Parameter<D, T, M>) -> Self {
        let adaptor = GlobalAdaptor::new(1.0, T::zero(), V::one());

        Self { adaptor, parameter }
    }

    pub fn initial_scale(&self, scale: f64) -> Self {
        let adaptor = self.adaptor.initial_scale(scale);

        let new_self = self.clone();
        Self {
            adaptor,
            ..new_self
        }
    }
}

impl<'a, D, T, M, V> SRWMBuilder<'a, D, T, M, V>
where
    D: Rv<T> + Mean<T> + Variance<V> + Clone,
    T: Clone,
    V: Clone,
{
    pub fn new_from_prior(parameter: &'a Parameter<D, T, M>) -> Option<Self> {
        let prior_variance = parameter.prior.variance()?;
        let prior_mean = parameter.prior.mean()?;

        let adaptor = GlobalAdaptor::new(0.0, prior_mean, prior_variance);

        Some(Self { parameter, adaptor })
    }
}

impl<'a, D, T, M, V> SRWMBuilder<'a, D, T, M, V>
where
    D: Rv<T> + Clone,
    T: Clone,
    V: Clone,
{
    pub fn new_with_mean_and_variance(
        parameter: &'a Parameter<D, T, M>,
        mean: T,
        variance: V,
    ) -> Self {
        let adaptor = GlobalAdaptor::new(0.0, mean, variance);

        Self { adaptor, parameter }
    }

    pub fn proposal_scale(&mut self, scale: f64) {
        self.adaptor.proposal_scale = scale;
    }
}

impl<'a, D, T, M, R, L> StepperBuilder<M, R, L>
    for SRWMBuilder<'a, D, T, M, T>
where
    R: 'static + Rng + Clone + Send + Sync,
    M: 'static + Clone + Sync + Send,
    D: 'static + Rv<T> + Clone + Send + Sync,
    T: 'static + Float + Into<f64> + From<f64> + Clone + Send + Sync,
    L: 'static + Fn(&M) -> f64 + Clone + Sync + Send,
{
    fn build<'b>(
        &self,
        rng: &'b mut R,
        log_likelihood: &'b L,
    ) -> Box<dyn SteppingAlg<M> + 'b> {
        Box::new(SRWM::new(self.parameter, log_likelihood, rng, self.adaptor.clone()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::steppers::{StepperBuilder, SteppingAlg};
    use crate::Lens;
    use rand::rngs::mock::StepRng;
    use rv::dist::Gaussian;

    #[derive(Clone)]
    struct Model {
        x: f64,
    }

    #[test]
    fn new_from_prior() {
        let parameter = Parameter::new(
            "x".to_owned(),
            Gaussian::standard(),
            make_lens!(Model, f64, x),
        );

        let builder = SRWMBuilder::new_from_prior(&parameter)
            .expect("Couldn't unwrap SRWMBuilder");

        assert_eq!(builder.adaptor.proposal_scale, 0.0);
        assert_eq!(builder.adaptor.mu, 0.0);
        assert_eq!(builder.adaptor.scale, 1.0);
    }

    #[test]
    fn new_from_given() {
        let parameter = Parameter::new(
            "x".to_owned(),
            Gaussian::standard(),
            make_lens!(Model, f64, x),
        );

        let builder =
            SRWMBuilder::new_with_mean_and_variance(&parameter, 1.0, 2.0);

        assert_eq!(builder.adaptor.proposal_scale, 0.0);
        assert_eq!(builder.adaptor.mu, 1.0);
        assert_eq!(builder.adaptor.scale, 2.0);
    }

    #[test]
    fn builds_correctly() {
        let parameter: Parameter<Gaussian, f64, Model> = Parameter::new(
            "x".to_owned(),
            Gaussian::standard(),
            make_lens!(Model, f64, x),
        );

        let builder: SRWMBuilder<'_, Gaussian, f64, Model, f64> =
            SRWMBuilder::new_with_mean_and_variance(&parameter, 1.0, 2.0);

        let ll = |m: &Model| m.x;

        let mut rng = StepRng::new(0, 1);
        let mut sampler = builder.build(&mut rng, &ll);
        let next_model = sampler.step(Model { x: 0.0 });
        println!("{}", parameter.lens.get(&next_model));
    }
}
