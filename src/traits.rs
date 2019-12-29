//! Traits!
//!

use num::*;
use nalgebra::Scalar;

/// Trait for scalar types in RMCMC
pub trait ScalarType: Clone + Send + Sync 
                    + Into<f64>
                    + Scalar 
                    + Float + FromPrimitive + ToPrimitive
{}

impl ScalarType for f32 {}
impl ScalarType for f64 {}


/// Extension Type for DiscreteSRWM supported types
pub trait DiscreteType: Integer + Saturating + ToPrimitive + FromPrimitive + Clone + Send + Sync  {}
impl DiscreteType for u8 {}
impl DiscreteType for u16 {}
impl DiscreteType for u32 {}
impl DiscreteType for u64 {}
impl DiscreteType for i8 {}
impl DiscreteType for i16 {}
impl DiscreteType for i32 {}
impl DiscreteType for i64 {}
