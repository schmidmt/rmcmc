//! Kameleon MCMC

// use std::format;

extern crate rand;
use rand::Rng;
use rand::distributions::Uniform;
use rand::seq::index;
use std::cmp;

extern crate num_traits;
use self::num_traits::sign::Signed;

extern crate nalgebra;
use self::nalgebra::base::*;
use self::nalgebra::base::storage::Storage;
// use self::nalgebra::base::dimension::U1;

use rv::traits::{Mean, Rv, Variance};

use parameter::Parameter;
use steppers::{SteppingAlg, AnnealingAlg};

use std::marker::PhantomData;

#[derive(Debug)]
struct KameleonAdaptor<N, D, S> 
where
    N: Scalar + PartialOrd + Signed,
    D: Dim,
    S: Storage<N, D> + Clone,
{
    // history of values
    history: Vec<Vector<N, D, S>>,
    subset_indicies: index::IndexVec,
    subset_size: usize,
    // scaling parameter nu
    adapt_nu: bool,
    nu: f64,
    // scaling parameter gamma
    adapt_gamma: bool,
    gamma: f64,
    // Number of current adaptation step.
    step: usize,
    // Alpha to stochastically optimize towards
    target_alpha: f64,
    // enable updates or not
    enabled: bool,
}

impl<N, D, S> KameleonAdaptor<N, D, S>
where
    N: Scalar + PartialOrd + Signed,
    D: Dim,
    S: Storage<N, D> + Clone
{
    pub fn new(initial_gamma: f64, adapt_gamma: bool, initial_nu: f64, adapt_nu: bool, subset_size: usize) -> Self {
        let history = Vec::new();
        let subset_indicies: index::IndexVec = index::IndexVec::from(vec![0 as usize]);

        KameleonAdaptor {
            history: history.to_owned(),
            subset_indicies: subset_indicies.to_owned(),
            subset_size,
            adapt_nu,
            nu: initial_nu,
            adapt_gamma,
            gamma: initial_gamma,
            step: 0,
            target_alpha: 0.234,
            enabled: true,
        }
    }

    fn enable(&mut self) {
        self.enabled = true
    }

    fn disable(&mut self) {
        self.enabled = false;
    }

    fn update<R: Rng>(&mut self, rng: &mut R, alpha: f64, new_value: Vector<N, D, S>) {
        if self.enabled {
            self.step += 1;
            let g = 0.9 / (self.step as f64).powf(0.9);
            let delta = alpha - self.target_alpha;
            if self.adapt_nu {
                self.nu = self.nu + g * delta;
            }
            
            // resample the subset of points
            let p_draw_new = 1.0 / (self.step as f64).sqrt();
            let p = rng.sample(Uniform::new(0.0, 1.0));
            if p < p_draw_new {
                self.subset_indicies = index::sample(
                    rng,
                    self.history.len(),
                    cmp::min(self.history.len(), self.subset_size)
                )
            }
        }
    }
}




/// MCMC Kameleon 
/// 
/// R: Rv with Mean and Variance traits
/// N: Scalar Type
/// D: Nimensionality
/// S: Storage Type
/// L: Likelihood Type
pub struct Kameleon<R, N, D, S, SV, M, L>
where
    R: Rv<Vector<N, D, S>> + Variance<SquareMatrix<N, D, SV>> + Mean<Vector<N, D, S>> + Clone,
    N: Scalar + PartialOrd + Signed,
    D: Dim,
    S: Storage<N, D>,
    SV: Storage<N, D, D>,
    M: Clone + 'static,
    L: Fn(&M) -> f64 + Sync + Clone
{
    pub parameter: Parameter<R, Vector<N, D, S>, M>,
    pub loglikelihood: L,
    pub current_score: Option<f64>,
    pub temperature: f64,
    pub log_acceptance: f64,
    pub sv_phantom: PhantomData<SV>,
}

impl<R, N, D, S, SV, M, L> Kameleon<R, N, D, S, SV, M, L>
where
    R: Rv<Vector<N, D, S>> + Variance<SquareMatrix<N, D, SV>> + Mean<Vector<N, D, S>> + Clone,
    N: Scalar + PartialOrd + Signed,
    D: Dim,
    S: Storage<N, D>,
    SV: Storage<N, D, D>,
    M: Clone + 'static,
    L: Fn(&M) -> f64 + Sync + Clone
{
    pub fn new(
        parameter: Parameter<R, Vector<N, D, S>, M>,
        loglikelihood: L
    ) -> Self {
        Kameleon {
            parameter,
            loglikelihood,
            current_score: None,
            temperature: 1.0,
            log_acceptance: 0.0,
            sv_phantom: PhantomData,
        }
    }
}


macro_rules! impl_traits {
    ($kind: ty) => {
        impl<R, D, S, SV, M, L> SteppingAlg<'a, M, R> for Kameleon<R, $kind, D, S, SV, M, L>
        where
            R: Rv<Vector<$kind, D, S>> + Variance<SquareMatrix<$kind, D, SV>> + Mean<Vector<$kind, D, S>> + Clone,
            D: Dim,
            S: Storage<$kind, D>,
            SV: Storage<$kind, D, D>,
            M: Clone + 'static,
            L: Fn(&M) -> f64 + Sync + Clone
        {
            fn step<Rn: Rng>(&self, rng: &mut Rn, model: &mut M) -> Self { std::unimplemented!(); }
            fn adapt_on(&self) -> Self { std::unimplemented!(); }
            fn adapt_off(&self) -> Self { std::unimplemented!(); }
        }

        impl<R, D, S, SV, M, L> Clone for Kameleon<R, $kind, D, S, SV, M, L>
            where
                R: Rv<Vector<$kind, D, S>> + Variance<SquareMatrix<$kind, D, SV>> + Mean<Vector<$kind, D, S>> + Clone,
                D: Dim,
                S: Storage<$kind, D>,
                SV: Storage<$kind, D, D>,
                M: Clone + 'static,
                L: Fn(&M) -> f64 + Sync + Clone
        {
            fn clone(&self) -> Self {
                Kameleon {
                    parameter: self.parameter.clone(),
                    loglikelihood: self.loglikelihood.clone(),
                    current_score: self.current_score,
                    temperature: self.temperature,
                    log_acceptance: self.log_acceptance,
                    sv_phantom: self.sv_phantom,
                }
            }
        }

        impl<R, D, S, SV, M, L> AnnealingAlg<M> for Kameleon<R, $kind, D, S, SV, M, L>
        where
            R: Rv<Vector<$kind, D, S>> + Variance<SquareMatrix<$kind, D, SV>> + Mean<Vector<$kind, D, S>> + Clone,
            D: Dim,
            S: Storage<$kind, D>,
            SV: Storage<$kind, D, D>,
            M: Clone + 'static,
            L: Fn(&M) -> f64 + Sync + Clone
         {
            fn set_temperature(&self, t: f64) -> Self {
                Kameleon {
                    temperature: t,
                    ..(*self).clone()
                }
            }
         }
    };
}

impl_traits!(f64);
impl_traits!(f32);
