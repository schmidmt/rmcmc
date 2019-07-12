//! Likelihood Containers for gradient free and gradient full likelihoods

/*
use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::Dim;
use nalgebra::base::VectorN;
*/
use std::fmt;

/// Likelihood Calculation without Gradient
pub trait Likelihood<M>: Sync + Clone + fmt::Debug {
    /// log likelihood of the given model
    fn ln_f(&self, model: &M) -> f64;
}

/*
/// Likelihood Calculation with Gradient
pub trait LikelihoodWithGradient<M>: Likelihood<M>
where
    DefaultAllocator: Allocator<f64, Self::D>,
{
    /// Dimension of gradient
    type D: Dim;
    /// Gradient calculation for given model value
    fn grad_ln_f(&self, model: &M) -> VectorN<f64, Self::D>;
}
*/
