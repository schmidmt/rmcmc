use nalgebra::base::dimension::Dim; 
use nalgebra::base::VectorN;
use nalgebra::DefaultAllocator;
use nalgebra::allocator::Allocator;
use std::fmt;
use common::Model;


/// Likelihood Calculation
pub trait Likelihood<M: Model>: Sync + Clone + fmt::Debug {
    fn ln_f(&self, model: &M) -> f64;
}

/// Likelihood Calculation with Gradient
pub trait LikelihoodWithGradient<M: Model>: Likelihood<M>
where
    DefaultAllocator: Allocator<f64, Self::D>
{
    type D: Dim;
    fn grad_ln_f(&self, model: &M) -> VectorN<f64, Self::D>;
}
