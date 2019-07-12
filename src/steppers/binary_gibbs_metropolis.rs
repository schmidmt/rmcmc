//! # Binary Gibbs Metropolis Algorithm
//! Preforms sampling on binary random variables.


use std::fmt::Debug;
extern crate rand;
use rand::Rng;
use rand::seq::SliceRandom;

use rv::traits::Rv;
use parameter::Parameter;

use steppers::{SteppingAlg, AdaptationStatus, AdaptationMode, util, ModelAndLikelihood};
use statistics::Statistic;

#[derive(Clone, Debug)]
pub struct BinaryGibbsMetropolis<D, T, M, L>
where
    D: Rv<T> + Clone + Debug,
    M: 'static + Clone + Debug,
    L: Fn(&M) -> f64 + Clone + Sync
{
    pub parameter: Parameter<D, T, M>,
    pub log_likelihood: L,
    pub current_score: Option<f64>,
    pub order: Vec<usize>,
    pub transit_p: f64,
}

impl<D, M, L> BinaryGibbsMetropolis<D, Vec<bool>, M, L>
where
    D: Rv<Vec<bool>> + Clone + Debug,
    M: 'static + Clone + Debug,
    L: Fn(&M) -> f64 + Clone + Sync
{
    pub fn new(
        parameter: Parameter<D, Vec<bool>, M>,
        log_likelihood: L,
        init_model: &M,
        transit_p: f64
    ) -> Option<Self> {

        let size = parameter.lens.get(init_model).len();

        Some(Self {
            parameter,
            log_likelihood,
            current_score: None,
            order: (0..size).collect(),
            transit_p,
        })
    }
}


impl<D, M, L, R> SteppingAlg<'a, M, R> for BinaryGibbsMetropolis<D, Vec<bool>, M, L>
where
    D: Rv<Vec<bool>> + Clone + Debug,
    M: 'static + Clone + Debug,
    L: Fn(&M) -> f64 + Clone + Sync + Debug,
    R: Rng,
{
    fn set_adapt(&mut self, _mode: AdaptationMode) {}
    fn get_adapt(&self) -> AdaptationStatus {
        AdaptationStatus::Disabled
    }
    fn get_statistics(&self) -> Vec<Statistic<M, R>> {
        Vec::new()
    }
    fn reset(&mut self) {}

    fn step(&mut self, rng: &mut R, model: M) -> M {
        self.order.shuffle(rng);
        let mut log_p = match self.current_score {
            Some(p) => p,
            None => (self.log_likelihood)(&model)
        };
        let mut m = model.clone();
        let mut bool_vec = self.parameter.lens.get(&m.clone());

        self.order.iter().for_each(|idx| {
            if rng.gen::<f64>() < self.transit_p {
                bool_vec[*idx] = !bool_vec[*idx];
                self.parameter.lens.set_in_place(&mut m, bool_vec.clone());
                let log_p_candidate = (self.log_likelihood)(&m);
                match util::metropolis_select(rng, log_p_candidate - log_p, bool_vec[*idx], !bool_vec[*idx]) {
                    util::MetroplisUpdate::Accepted(q, _) => {
                        bool_vec[*idx] = q;
                        log_p = log_p_candidate;
                    },
                    util::MetroplisUpdate::Rejected(q, _) => {
                        bool_vec[*idx] = q;
                    }
                }
            }
        });

        self.current_score = Some(log_p);
        self.parameter.lens.set_in_place(&mut m, bool_vec);
        m
    }
}
