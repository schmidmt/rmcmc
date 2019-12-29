//! Helpful utilities

use nalgebra::{DVector, DMatrix, Scalar, Dynamic};
use nalgebra::linalg::{SVD, Cholesky, SymmetricEigen};
use alga::general::RealField;
use std::f64;
use std::cmp::Ordering;

mod likelihood;
pub use likelihood::*;

#[cfg(test)]
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

/// Response struct for nearest_spd
#[derive(Clone, Debug)]
pub struct NearestSPD {
    /// Nearest Matrix
    pub m: DMatrix<f64>,
    /// Cholesky Decomposition of m
    pub cholesky: Cholesky<f64, Dynamic>
}

impl NearestSPD {
    /// Find the nearest Symmetric Positive (Semi-)Definite matrix
    ///
    /// # Example
    /// ```rust
    /// use nalgebra::DMatrix;
    /// use rmcmc::utils::NearestSPD;
    ///
    /// let m = DMatrix::from_row_slice(3, 3, &[
    ///     0.0, 0.0, 0.0,
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0
    /// ]);
    ///
    /// let nearest = NearestSPD::nearest(&m)
    ///     .expect("Should have computed nearest value");
    ///
    /// let expected = DMatrix::from_row_slice(3, 3, &[
    ///     0.1768, 0.2500, 0.1768,
    ///     0.2500, 0.3536, 0.2500,
    ///     0.1768, 0.2500, 0.1768
    /// ]);
    ///
    /// assert!(expected.relative_eq(&nearest.m, 1E-4, 1E-4));
    /// ```
    pub fn nearest(m: &DMatrix<f64>) -> Option<Self> {
        const MAX_ITER: usize = 100;
        const EPS: f64 = 1E-10;

        let b = (m + m.transpose()).map(|x| x / 2.0);
        let b_svd = SVD::new(b.clone(), false, true);

        let s = b_svd.singular_values;
        let v = b_svd.v_t.unwrap();
        let h = v.transpose() * (DMatrix::from_diagonal(&s) * &v);
        let ahat_temp = (&b + &h).map(|x| x / 2.0);
        let mut ahat = (&ahat_temp + &ahat_temp.transpose()).map(|x| x / 2.0);

        for k in 0..MAX_ITER {
            let eigs = SymmetricEigen::new(ahat.clone());
            let min_eig_value = eigs.eigenvalues.iter()
                .fold(f64::INFINITY, |a, &b| {
                    match PartialOrd::partial_cmp(&a, &b) {
                        None => f64::NAN,
                        Some(Ordering::Less) => a,
                        _ => b,
                    }
                });
            let chol_try = ahat.clone().cholesky();

            if min_eig_value < 0.0 || chol_try.is_none() {
                let k = k as f64;
                ahat = ahat.clone() + DMatrix::from_diagonal_element(ahat.nrows(), ahat.ncols(), -min_eig_value * k * k + EPS)
            } else {
                return Some(
                    Self {
                        m: ahat,
                        cholesky: chol_try.unwrap(),
                    }
                )
            }
        }
        None
    }
}
