//! # Metropolis Hastings Utilities

use rand::Rng;

/// Status given to a Metropolis update
#[derive(Clone, Debug)]
pub enum MetropolisUpdate<'a, M>
where
    M: 'a + Clone,
{
    /// The update was accepted
    Accepted(&'a M, f64),
    /// The update was rejected
    Rejected(&'a M, f64),
}

/// Metropolis Update
/// Given a symmetric proposal distribution, this function will update proportional to the
/// likelihood.
///
/// # Parameters
/// * `rng` - random number generator
/// * `log_likelihood_delta` - Difference between current and proposed log_likelihoods.
/// * `proposed` - Candidate new model
/// * `current` - Current value
pub fn metropolis_select<'a, M: Clone, R: Rng>(
    rng: &mut R,
    log_likelihood_delta: f64,
    proposed: &'a M,
    current: &'a M,
) -> MetropolisUpdate<'a, M> {
    assert!(
        !log_likelihood_delta.is_nan(),
        "metropolis_select cannot be given a log_likelihood_delta of NAN"
    );
    let lll = log_likelihood_delta.min(0.0);

    if lll == 0.0 || rng.gen::<f64>().ln() < log_likelihood_delta {
        MetropolisUpdate::Accepted(proposed, lll)
    } else {
        MetropolisUpdate::Rejected(current, lll)
    }
}
