/*
use rv::traits::Rv;
/// Generate a log_likelihood closure with a given set of data
///
/// # Example
/// ```rust
/// use rmcmc::utils::log_likelihood_from_data;
/// use rv::dist::Poisson;
/// use assert::close;
///
/// let data: Vec<u32> = vec![2, 3, 2, 1, 1, 3, 2, 4, 3, 4];
///
/// let log_likelihood = log_likelihood_from_data(&data, |&mean| { Poisson::new(mean) });
///
/// close(log_likelihood(&3_f64), -16.34552039335715, 1E-10);
/// ```
pub fn log_likelihood_from_data<'a, T, M, F, X>(
    data: &'a [T],
    model_to_dist: F,
) -> impl Fn(&M) -> f64 + Clone + 'a
where
    X: Rv<T>,
    F: Fn(&M) -> rv::Result<X> + Clone + 'a,
{
    move |model| match model_to_dist(model) {
        Ok(dist) => data.iter().fold(0.0, |ll, d| ll + dist.ln_f(d)),
        Err(_) => std::f64::NEG_INFINITY,
    }
}
*/
