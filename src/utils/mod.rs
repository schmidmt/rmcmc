//! Helpful utilities

use nalgebra::{DVector, DMatrix, Scalar};
use alga::general::RealField;

mod likelihood;
pub use likelihood::*;

// #[cfg(test)]
pub mod test;

/// Perform outer product on two DVectors
#[inline]
pub fn outer<N: Scalar + RealField>(left: &DVector<N>, right: &DVector<N>) -> DMatrix<N> {
    let mut res = unsafe { DMatrix::new_uninitialized(left.len(), right.len()) };

    for i in 0 .. left.len() {
        for j in 0 .. right.len() {
            unsafe {
                *res.get_unchecked_mut((i, j)) = *left.get_unchecked(i) * *right.get_unchecked(j);
            }
        }
    }

    res
}
