use crate::steppers::{AdaptationMode, SteppingAlg};

#[doc(hidden)]
pub fn draw_from_stepper<'a, M, A>(
    stepper: A,
    init: M,
    n_draws: usize,
    n_warmup: usize,
    thinning: usize,
    keep_warmup: bool,
) -> Vec<M>
where
    M: Clone + Sync + Send,
    A: SteppingAlg<M> + Send + Sync,
{
    let mut stepper = stepper;

    let prior_sample = stepper.prior_draw(init);

    // WarmUp
    stepper.set_adapt(AdaptationMode::Enabled);

    let mut warmup_draws = if keep_warmup {
        (0..n_warmup)
            .scan(prior_sample.clone(), |m, _| {
                *m = stepper.step((*m).clone());
                Some(m.clone())
            })
            .collect()
    } else {
        let mp =
            (0..n_warmup).fold(prior_sample.clone(), |m, _| stepper.step(m));
        vec![mp]
    };

    // Draw the steps from the chain
    stepper.set_adapt(AdaptationMode::Disabled);

    let warmed_model: M = if warmup_draws.is_empty() {
        prior_sample
    } else {
        warmup_draws.last().unwrap().clone()
    };

    let draws: Vec<M> = (0..(n_draws * thinning))
        .scan(warmed_model, |m, _| {
            *m = stepper.step((*m).clone());
            Some(m.clone())
        })
        .step_by(thinning)
        .collect();

    if keep_warmup {
        warmup_draws.extend(draws);
        warmup_draws
    } else {
        draws
    }
}

#[cfg(test)]
mod test {
    /*
    use super::*;
    use rand::SeedableRng;
    use std::sync::{Arc, RwLock};
    const SEED: [u8; 32] = [0; 32];
    */
}
