use rand::Rng;
use std::marker::PhantomData;
use steppers::*;
use reduce::Reduce;

/// Stepper Group
pub struct Group<M, R>
where
    M: Clone,
    R: Rng,
{
    steppers: Vec<Box<SteppingAlg<M, R>>>,
    current_likelihood: Option<f64>,
    phantom_m: PhantomData<M>,
    phantom_r: PhantomData<R>
}

impl<M, R> Clone for Group<M, R>
where
    M: 'static + Clone,
    R: 'static + Rng,
{
    fn clone(&self) -> Self {
        Self {
            steppers: self.steppers.iter().map(|x| (*x).box_clone()).collect(),
            current_likelihood: self.current_likelihood,
            phantom_m: PhantomData,
            phantom_r: PhantomData,
        }
    }
}

unsafe impl<M, R> Sync for Group<M, R>
where
    M: Clone,
    R: Rng,
{}

unsafe impl<M, R> Send for Group<M, R>
where
    M: Clone,
    R: Rng,
{}


impl<M, R: Rng> Group<M, R>
where
    M: Clone,
    R: Rng,
{
    pub fn new(steppers: Vec<Box<SteppingAlg<M, R>>>) -> Self {
        Group {
            steppers,
            current_likelihood: None,
            phantom_m: PhantomData,
            phantom_r: PhantomData
        }
    }
}

impl<M, R> SteppingAlg<M, R> for Group<M, R>
where
    M: 'static + Clone,
    R: 'static + Rng
{
    fn step(&mut self, rng: &mut R, model: M) -> M {
        let current_loglikelihood = self.current_likelihood;
        self.step_with_loglikelihood(rng, model, current_loglikelihood).model
    }

    fn step_with_loglikelihood(&mut self, rng: &mut R, model: M, loglikelihood: Option<f64>) -> ModelAndLikelihood<M> {
        let next_mll = self.steppers
            .iter_mut()
            .fold(ModelAndLikelihood::new(model, loglikelihood), |mll, stepper| stepper.step_with_loglikelihood(rng, mll.model, mll.loglikelihood));
        self.current_likelihood = next_mll.loglikelihood;
        next_mll
    }

    fn set_adapt(&mut self, mode: AdaptationMode) {
        self
            .steppers
            .iter_mut()
            .for_each(|s| s.set_adapt(mode))
    }

    fn get_adapt(&self) -> AdaptationStatus {
        self
            .steppers
            .iter()
            .map(|s| s.get_adapt())
            .reduce(|a, b| match (a, b) {
                (AdaptationStatus::Enabled, AdaptationStatus::Enabled) => AdaptationStatus::Enabled,
                (AdaptationStatus::Disabled, AdaptationStatus::Disabled) => AdaptationStatus::Disabled,
                _ => AdaptationStatus::Mixed
            })
            .unwrap_or(AdaptationStatus::Mixed)
    }

    fn reset(&mut self) {
        self
            .steppers
            .iter_mut()
            .for_each(|s| s.reset())
    } 

    fn box_clone(&self) -> Box<SteppingAlg<M, R>> {
        Box::new((*self).clone())
    }

    fn prior_draw(&self, rng: &mut R, model: M) -> M {
        self.steppers.iter().fold(model, |acc, s| s.prior_draw(rng, acc))
    }
}


#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    extern crate test;
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use nalgebra::{DMatrix, DVector, Vector, Vector2, Matrix2};
    use nalgebra::inverse;
    use rv::dist::{MvGaussian, Gaussian};
    use rv::prelude::*;
    use rv::misc::ks_test;
    use ::steppers::SRWM;
    use ::parameter::Parameter;
    use ::lens::*;
    use ::runner::Runner;
    use reduce::Reduce;

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Model1D {
        i: i32
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Model2D(DVector<f64>);


    macro_rules! vector_lens {
        ($idx:expr) => {
            Lens::new(
                |m: &Model2D| m.0[$idx],
                |m: &Model2D, t: f64| {
                    let mut c = m.0.clone();
                    c[$idx] = t;
                    Model2D(c)
                }
            )
        };
    }

    fn vec_vec_mean(vecs: &Vec<DVector<f64>>) -> DVector<f64> {
        let mut acc = DVector::zeros(2);
        for vec in vecs {
            acc = acc + vec;
        }
        acc / (vecs.len() as f64)
    }
/*
    #[test]
    fn two_d_srwm_gaussian() {
        let mut rng: StdRng = SeedableRng::seed_from_u64(0);
        let lens_x = vector_lens!(0);
        let lens_y = vector_lens!(1);

        let generating_mu: DVector<f64> = DVector::repeat(2, 1.0);
        let generating_scale: DMatrix<f64> = DMatrix::identity(2, 2);

        let generating_dist = MvGaussian::new(
            generating_mu.clone(),
            generating_scale.clone()
        ).unwrap();
        let n_data = 100;
        let data: Vec<DVector<f64>> = generating_dist.sample(n_data, &mut rng);
        let data_mean: DVector<f64> = vec_vec_mean(&data);

        let prior_mean = DVector::zeros(2);
        let prior_variance = DMatrix::identity(2, 2);

        let parameter_x = Parameter::new(
            "x".to_string(),
            Gaussian::new(prior_mean[0], prior_variance[(0, 0)]).unwrap(),
            lens_x.clone()
        );

        let parameter_y = Parameter::new(
            "y".to_string(),
            Gaussian::new(prior_mean[1], prior_variance[(1, 1)]).unwrap(),
            lens_y.clone()
        );

        let generating_scale_2 = generating_scale.clone();
        let loglikelihood = move |m: &Model2D| {
            let mvg = MvGaussian::new((m.clone()).0, generating_scale_2.clone()).unwrap();
            data.iter().fold(0.0, |acc, d| acc + mvg.ln_f(&d))
        };

        let alg_start = Group::new(vec![
            Box::new(SRWM::new(parameter_x, loglikelihood.clone(), Some(1.0)).unwrap()),
            Box::new(SRWM::new(parameter_y, loglikelihood, Some(1.0)).unwrap()),
        ]);

        let m = Model2D(DVector::zeros(2));

        let runner = Runner::new(alg_start.clone())
            .warmup(0)
            .chains(1)
            .thinning(10);

        let results: Vec<Vec<Model2D>> = runner.run(&mut rng, m);
        let samples: Vec<DVector<f64>> = results
            .iter()
            .map(|chain| -> Vec<DVector<f64>> {
                chain.iter().map(|g| g.clone().0).collect()
            })
            .flatten()
            .collect();

        let (mcmc_x_draws, mcmc_y_draw): (Vec<f64>, Vec<f64>) = results
            .iter()
            .flatten()
            .map(|m| (lens_x.get(m), lens_y.get(m)))
            .unzip();

        let n = n_data as f64;
        let prior_variance_inv = prior_variance.qr().try_inverse().unwrap();
        let generating_variance_inv = generating_scale.clone().qr().try_inverse().unwrap();
        let posterior_variance = (prior_variance_inv.clone() + n * generating_variance_inv.clone()).try_inverse().unwrap();
        let posterior_mean = posterior_variance.clone() * (prior_variance_inv * prior_mean + n * generating_variance_inv * data_mean);

        let posterior_dist = MvGaussian::new(posterior_mean, posterior_variance).unwrap();
        let (posterior_x_draws, posterior_y_draws): (Vec<f64>, Vec<f64>) = posterior_dist
            .sample(1000, &mut rng)
            .iter()
            .map(|v| (v[0], v[1]))
            .unzip();

        let (_, p_x) = ks_test();
    }
*/


/*
    #[use_mocks]
    #[test]
    fn group_calls_steppers_correctly() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut m1 = new_mock!(SteppingAlg<Model, StdRng>);
        let mut m2 = new_mock!(SteppingAlg<Model, StdRng>);

        given! {
            <m1 as SteppingAlg<Model, StdRng>>::{
                step_with_loglikelihood(any_value(), eq(Model { i : 0 }), eq(None)) then_return_from |_| (Model { i: 1 }, Some(0.1)) times 1; 
                step_with_loglikelihood(any_value(), eq(Model { i : 2 }), eq(Some(0.5))) then_return_from |_| (Model { i: 3 }, Some(0.3)) times 1; 
            };

            <m2 as SteppingAlg<Model, StdRng>>::{
                step_with_loglikelihood(any_value(), eq(Model { i: 1}), eq(Some(0.1))) then_return_from |_| (Model { i: 2 }, Some(0.5)) times 1; 
                step_with_loglikelihood(any_value(), eq(Model { i: 3}), eq(Some(0.3))) then_return_from |_| (Model { i: 4 }, Some(0.0)) times 1; 
            };
        }

        expect_interactions! {
            <m1 as SteppingAlg<Model, StdRng>>::step_with_loglikelihood(any_value(), eq(Model { i : 0 }), eq(None)) times 1;
            <m1 as SteppingAlg<Model, StdRng>>::step_with_loglikelihood(any_value(), eq(Model { i : 2 }), eq(Some(0.5))) times 1;

            <m2 as SteppingAlg<Model, StdRng>>::step_with_loglikelihood(any_value(), eq(Model { i: 1 }), eq(Some(0.1))) times 1;
            <m2 as SteppingAlg<Model, StdRng>>::step_with_loglikelihood(any_value(), eq(Model { i: 3 }), eq(Some(0.3))) times 1;
        }

        let mut group = Group::new(vec![Box::new(m1), Box::new(m2)]);
        let model_0 = Model { i: 0 };
        let model_1 = group.step(&mut rng, model_0);
        assert_eq!(model_1, Model { i: 2 });
        assert_eq!(group.current_likelihood, Some(0.5));
        let model_2 = group.step(&mut rng, model_1);
        assert_eq!(model_2, Model { i: 4 });
        assert_eq!(group.current_likelihood, Some(0.0));
    }

    #[use_mocks]
    #[test]
    fn reset_is_broadcast_to_steppers() {
        let mut mock_sampler_1 = new_mock!(SteppingAlg<Model, StdRng>);
        let mut mock_sampler_2 = new_mock!(SteppingAlg<Model, StdRng>);

        given! {
            <mock_sampler_1 as SteppingAlg<Model, StdRng>>::reset() then_return () always;
            <mock_sampler_2 as SteppingAlg<Model, StdRng>>::reset() then_return () always;
        }

        expect_interactions! {
            <mock_sampler_1 as SteppingAlg<Model, StdRng>>::reset() times 1;
            <mock_sampler_2 as SteppingAlg<Model, StdRng>>::reset() times 1;
        }

        let mut group = Group::new(vec![Box::new(mock_sampler_1), Box::new(mock_sampler_2)]);
        group.reset();
    }

    #[use_mocks]
    #[test]
    fn set_adapt_is_broadcast_to_steppers() {
        let mut mock_sampler_1 = new_mock!(SteppingAlg<Model, StdRng>);
        let mut mock_sampler_2 = new_mock!(SteppingAlg<Model, StdRng>);

        given! {
            <mock_sampler_1 as SteppingAlg<Model, StdRng>>::set_adapt(any_value()) then_return () always;
            <mock_sampler_2 as SteppingAlg<Model, StdRng>>::set_adapt(any_value()) then_return () always;
        }

        expect_interactions! {
            <mock_sampler_1 as SteppingAlg<Model, StdRng>>::set_adapt(eq(AdaptationMode::Enabled)) times 1;
            <mock_sampler_2 as SteppingAlg<Model, StdRng>>::set_adapt(eq(AdaptationMode::Enabled)) times 1;
            <mock_sampler_1 as SteppingAlg<Model, StdRng>>::set_adapt(eq(AdaptationMode::Disabled)) times 1;
            <mock_sampler_2 as SteppingAlg<Model, StdRng>>::set_adapt(eq(AdaptationMode::Disabled)) times 1;
        }

        let mut group = Group::new(vec![Box::new(mock_sampler_1), Box::new(mock_sampler_2)]);
        group.set_adapt(AdaptationMode::Enabled);
        group.set_adapt(AdaptationMode::Disabled);
    }


    #[use_mocks]
    #[test]
    fn get_adapt_returns_correct_value() {
        let mut mock_sampler_enabled_1 = new_mock!(SteppingAlg<Model, StdRng>);
        let mut mock_sampler_enabled_2 = new_mock!(SteppingAlg<Model, StdRng>);
        let mut mock_sampler_disabled_1 = new_mock!(SteppingAlg<Model, StdRng>);
        let mut mock_sampler_disabled_2 = new_mock!(SteppingAlg<Model, StdRng>);
        let mut mock_sampler_mixed = new_mock!(SteppingAlg<Model, StdRng>);

        given! {
            <mock_sampler_enabled_1 as SteppingAlg<Model, StdRng>>::get_adapt() then_return AdaptationStatus::Enabled always;
            <mock_sampler_enabled_2 as SteppingAlg<Model, StdRng>>::get_adapt() then_return AdaptationStatus::Enabled always;
            <mock_sampler_disabled_1 as SteppingAlg<Model, StdRng>>::get_adapt() then_return AdaptationStatus::Disabled always;
            <mock_sampler_disabled_2 as SteppingAlg<Model, StdRng>>::get_adapt() then_return AdaptationStatus::Disabled always;
            <mock_sampler_mixed as SteppingAlg<Model, StdRng>>::get_adapt() then_return AdaptationStatus::Mixed always;
        }

        let group_enabled = Group::new(vec![Box::new(mock_sampler_enabled_1), Box::new(mock_sampler_enabled_2)]);
        assert_eq!(group_enabled.get_adapt(), AdaptationStatus::Enabled);

        let group_disabled = Group::new(vec![Box::new(mock_sampler_disabled_1), Box::new(mock_sampler_disabled_2)]);
        assert_eq!(group_disabled.get_adapt(), AdaptationStatus::Disabled);

        let group_mixed = Group::new(vec![Box::new(mock_sampler_mixed)]);
        assert_eq!(group_mixed.get_adapt(), AdaptationStatus::Mixed);

    }
*/
}
