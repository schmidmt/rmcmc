use rand::Rng;

/// Status given to a Metropolis update
#[derive(Clone, Debug)]
pub enum MetroplisUpdate<M>
where
    M: Clone
{
    /// The update was accepted
    Accepted(M, f64),
    /// The update was rejected
    Rejected(M, f64),
}

/// Metropolis Update
/// Given a symmetric proposal distribution, this function will update proportional to the
/// likelihood.
///
/// # Parameters
/// * `rng` random number generator
/// * `log_likelihood_delta` Difference between current and posposed log_likelihoods.
/// * `proposed` Candidate new model
/// * `current` Current value
pub fn metropolis_select<M: Clone, R: Rng>(
    rng: &mut R,
    log_likelihood_delta: f64,
    proposed: M,
    current: M
) -> MetroplisUpdate<M> {

    if rng.gen::<f64>().ln() < log_likelihood_delta {
        MetroplisUpdate::Accepted(proposed, log_likelihood_delta)
    } else {
        MetroplisUpdate::Rejected(current, log_likelihood_delta)
    }
}
