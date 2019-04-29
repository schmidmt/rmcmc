//! An implementation of the Global Adaptor

use steppers::adaptor::ScaleAdaptor;
use steppers::{AdaptationStatus, AdaptationMode};
use steppers::util::MetroplisUpdate;
/*
use nalgebra::base::{Vector, Matrix, Scalar, Dim};
use std::any::Any;
use std::fmt::Debug;
*/
use num::{FromPrimitive, Float};
use std::fmt;


/// # Globally Adaptive MC Adaptor
///
#[derive(Debug, Clone)]
pub struct GlobalAdaptor<T, V>
{
    // Scale factor *λ*
    log_lambda: f64,
    // Stochastic Mean *μ*
    mu: T,
    // Stochastic Scale *σ^2*
    scale: V,
    // Number of adaptation steps.
    step: usize,
    // Scale for proposals.
    pub proposal_scale: f64,
    // Initial proposal_scale for reset
    initial_proposal_scale: f64,
    // initial mean
    initial_mu: T,
    // initial scale
    initial_scale: V,
    // Alpha to stochastically optimize towards.
    target_alpha: f64,
    // Enables updates or not.
    enabled: bool,
}

impl<T, V> GlobalAdaptor<T, V>
where
    T: Clone,
    V: Clone
{
    pub fn new(initial_proposal_scale: f64, mean: T, scale: V) -> Self {

        GlobalAdaptor {
            log_lambda: 0.0,
            mu: mean.clone(),
            scale: scale.clone(),
            step: 0,
            proposal_scale: initial_proposal_scale,
            target_alpha: 0.234,
            enabled: false,
            initial_proposal_scale,
            initial_mu: mean,
            initial_scale: scale,
        }
    }
}

impl<T> ScaleAdaptor<T> for GlobalAdaptor<T, T>
where
    T: Float + fmt::Display + fmt::Debug
{
    fn get_scale(&self) -> f64 {
        self.proposal_scale
    }

    fn reset(&mut self) {
        self.log_lambda = 0.0;
        self.proposal_scale = self.initial_proposal_scale;
        self.step = 0;
        self.scale = self.initial_scale.clone();
        self.mu = self.initial_mu.clone();
        self.enabled = false;
    }

    fn set_mode(&mut self, mode: AdaptationMode) {
        match mode {
            AdaptationMode::Enabled => self.enabled = true,
            AdaptationMode::Disabled => self.enabled = false
        }
    }

    fn get_mode(&self) -> AdaptationStatus {
        if self.enabled {
            AdaptationStatus::Enabled
        } else {
            AdaptationStatus::Disabled
        }
    }

    fn update(&mut self, update: &MetroplisUpdate<T>) {
        if self.enabled {
            let (new_value, log_alpha) = match update {
                MetroplisUpdate::Accepted(x, y) => (x, y),
                MetroplisUpdate::Rejected(x, y) => (x, y),
            };
            let alpha = log_alpha.exp();
            let g: T = T::from(0.9 / f64::from_usize(self.step + 1).unwrap().powf(0.9)).unwrap();
            let delta: T = *new_value - self.mu;
            let new_log_lambda = self.log_lambda + g.to_f64().unwrap() * (alpha - self.target_alpha);
            let new_mu = self.mu + g * delta;
            let new_sigma = self.scale + g * ((delta * delta) - self.scale);
            let new_proposal_scale = T::from(new_log_lambda.exp()).unwrap() * new_sigma;

            assert!(
                new_proposal_scale > T::zero() && T::is_normal(new_proposal_scale),
                format!(
                    "update SrwmAdaptor = {:?}  \
                        (new_lambda = {}, new_sigma = {})
                        with alpha = {}, \
                        new_value = {} \
                        caused a bad proposal_scale...",
                    self,
                    new_log_lambda.exp(),
                    new_sigma,
                    alpha,
                    new_value
                )
            );

            self.log_lambda = new_log_lambda;
            self.mu = new_mu;
            self.scale = new_sigma;
            self.step += 1;
            self.proposal_scale = new_proposal_scale.to_f64().unwrap();
        }
    }
}

/*
impl<N, D, S> ScaleAdaptor<Vector<N, D, S>> for GlobalAdaptor<Vector<N, D, S>, Matrix<N, D, D, S>>
where
    N: Scalar + Copy + PartialEq + Debug + Any,
    D: Dim + Copy,
    S: Copy
{
    fn get_scale(&self) -> f64 {
        self.proposal_scale
    }

    fn reset(&mut self) {
        self.log_lambda = 0.0;
        self.proposal_scale = self.initial_proposal_scale;
        self.step = 0;
        self.scale = self.initial_scale.clone();
        self.mu = self.initial_mu.clone();
        self.enabled = false;
    }

    fn set_mode(&mut self, mode: AdaptationMode) {
        match mode {
            AdaptationMode::Enabled => self.enabled = true,
            AdaptationMode::Disabled => self.enabled = false
        }
    }

    fn get_mode(&self) -> AdaptationStatus {
        match self.enabled {
            true => AdaptationStatus::Enabled,
            false => AdaptationStatus::Disabled
        }
    }

    fn update(&mut self, update: &MetroplisUpdate<Vector<N, D, S>>) {

    }
}
*/

