use crate::utils::MeanAndVariance;
use itertools::Itertools;
use num::Float;

/// Gelman-Rubin Diagnostic R̂
/// See http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf
pub fn rhat<T: Float>(vals: Vec<Vec<T>>) -> T {
    // let vals: Vec<Vec<T>> = vals.iter().map::<Vec<T>, _>(|x| x.to_vec()).collect();
    let distinct_lengths: Vec<usize> =
        vals.iter().map(|x| x.len()).sorted().dedup().collect();
    assert_eq!(
        distinct_lengths.len(),
        1,
        "Unequal chain sizes! Cannot calculate rHat"
    );

    let m =
        T::from(vals.len()).expect("Cannot convert length of vector to type T");
    let n = T::from(distinct_lengths[0])
        .expect("Cannot convert length of vector to type T");

    let chain_mvs: Vec<MeanAndVariance<T>> = vals
        .iter()
        .map(|x| MeanAndVariance::from_values(x))
        .collect();
    let w = chain_mvs.iter().fold(T::zero(), |acc, x| acc + x.variance) / m;
    let theta_bar_bar =
        chain_mvs.iter().fold(T::zero(), |acc, x| acc + x.mean) / m;
    let b = n * chain_mvs
        .iter()
        .fold(T::zero(), |acc, x| acc + (x.mean - theta_bar_bar).powi(2))
        / (m - T::one());
    let var_hat_theta = (T::one() - T::one() / n) * w + b / n;
    (var_hat_theta / w).sqrt()
}
