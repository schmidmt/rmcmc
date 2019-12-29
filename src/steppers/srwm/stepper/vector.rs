use rand::Rng;
use rv::dist::MvGaussian;
use rv::traits::Rv;
use nalgebra::{DMatrix, DVector};

use crate::steppers::adaptors::{AdaptState, ScaleAdaptor, Adaptor};
use crate::steppers::helpers::metropolis_proposal;
use crate::steppers::helpers::MHStatus::*;
use crate::SteppingAlg;
use super::*;

impl<'a, Prior, Model, LogLikelihood, RNG> SteppingAlg<'a, Model, RNG>
    for SRWM<'a, Prior, DVector<f64>, DMatrix<f64>, Model, LogLikelihood, RNG>
where
    Model: Clone + Send + Sync,
    Prior: Rv<DVector<f64>> + Send + Sync,
    LogLikelihood: Fn(&Model) -> f64 + Send + Sync,
    RNG: Rng + Send + Sync,
{
    fn step(&mut self, rng: &mut RNG, model: Model) -> Model {
        let current_ll = self.current_ll_score;
        self.step_with_log_likelihood(rng, model, current_ll).0
    }

    fn step_with_log_likelihood(
        &mut self,
        rng: &mut RNG,
        model: Model,
        log_likelihood: Option<f64>,
    ) -> (Model, f64) {
        // Determine current state
        let current_value = self.parameter.lens().get(&model);
        let current_ll =
            log_likelihood.unwrap_or_else(|| (self.log_likelihood)(&model));

        let current_prior = self
            .current_prior_score
            .unwrap_or_else(|| self.parameter.prior(&model).ln_f(&current_value));

        let current_score = current_ll + current_prior;

        let proposal_dist =
            MvGaussian::new(current_value.map(|x| x.into()), self.adaptor.scale())
                .expect("Cannot create MvGaussain with given parameters");

        let proposed_value = proposal_dist.draw(rng).map(|x| x.into());
        let proposed_model =
            self.parameter.lens().set(model.clone(), proposed_value.clone());

        let proposed_prior = {
            let p = self.parameter.prior(&proposed_model).ln_f(&proposed_value);
            if p.is_nan() {
                std::f64::NEG_INFINITY
            } else {
                p
            }
        };

        let mut proposed_ll: Option<f64> = None;

        let proposed_score = if proposed_prior.is_finite() {
            let ll = (self.log_likelihood)(&proposed_model);
            proposed_ll = Some(ll);
            ll + proposed_prior
        } else {
            proposed_prior
        };

        // Do Metropolis Step

        let log_alpha = proposed_score - current_score;
        let update = metropolis_proposal(
            rng,
            log_alpha,
            &proposed_value,
            &current_value,
        );

        self.adaptor.update(&update);

        // Return appropriate value
        match update {
            Accepted(_, _) => {
                self.current_ll_score = proposed_ll;
                self.current_prior_score = Some(proposed_prior);
                (proposed_model, proposed_ll.unwrap())
            }
            Rejected(_, _) => (model, current_ll),
        }
    }
    fn draw_prior(&self, rng: &mut RNG, m: Model) -> Model {
        self.parameter.draw(m, rng)
    }

    fn adapt_enable(&mut self) {
        self.adaptor.enable();
    }

    fn adapt_disable(&mut self) {
        self.adaptor.disable();
    }

    fn adapt_state(&self) -> AdaptState {
        self.adaptor.state()
    }
}
