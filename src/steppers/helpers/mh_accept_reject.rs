use rand::Rng;
use std::fmt;
use std::fmt::{Debug, Formatter};

/// Metropolis-Hasting Accept / Reject Status
/// Contains the value of the next step (previous if rejected, proposal if accepted)
/// The enum values also contains likelihood of the given state
pub enum MHStatus<'a, M> {
    /// The proposed value was accepted
    Accepted(&'a M, f64),
    /// The proposed value was rejected
    Rejected(&'a M, f64),
}

impl<'a, M> Debug for MHStatus<'a, M>
where
    M: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use MHStatus::*;
        match *self {
            Accepted(m, l) => write!(f, "Accepted({:?}, {})", m, l.exp()),
            Rejected(m, l) => write!(f, "Rejected({:?}, {})", m, l.exp()),
        }
    }
}

/// Generate the next step in a Metropolis Accept/Reject Step given the difference in log_likelihoods
///
/// # Parameters
/// * `rng` - Random number generator
/// * `log_likelihood_delta` - Difference between current and proposed state log_likelihoods
/// * `proposed` - Proposed state
/// * `current` - Current State
pub fn metropolis_proposal<'a, RNG: Rng, M>(
    rng: &mut RNG,
    log_likelihood_delta: f64,
    proposed: &'a M,
    current: &'a M,
) -> MHStatus<'a, M> {
    let lll = log_likelihood_delta.min(0.0);

    if lll == 0.0 || rng.gen::<f64>().ln() < log_likelihood_delta {
        MHStatus::Accepted(proposed, lll)
    } else {
        MHStatus::Rejected(current, lll)
    }
}
