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
/// * `loglikelihood_delta` Difference between current and posposed loglikelihoods.
/// * `proposed` Candidate new model
/// * `current` Current value
pub fn metropolis_select<M: Clone, R: Rng>(
    rng: &mut R,
    loglikelihood_delta: f64,
    proposed: M,
    current: M
) -> MetroplisUpdate<M> {
    assert!(!loglikelihood_delta.is_nan(), "metropolis_select cannot be given a log_likelihood_delta of NAN");
    let lll = loglikelihood_delta.min(0.0);
    
    if lll == 0.0 || rng.gen::<f64>().ln() < loglikelihood_delta {
        MetroplisUpdate::Accepted(proposed, lll)
    } else {
        MetroplisUpdate::Rejected(current, lll)
    }
}
