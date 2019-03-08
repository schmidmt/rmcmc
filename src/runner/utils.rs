use steppers::{SteppingAlg, AdaptationMode};
use rand::prelude::*;
use std::sync::{Arc, RwLock};
use std::ops::DerefMut;

pub fn draw_from_stepper<M, A, R>(
    rng: Arc<RwLock<&mut R>>,
    stepper: A,
    init: M,
    n_draws: usize,
    n_warmup: usize,
    thinning: usize,
    keep_warmup: bool,
) -> Vec<M> 
where
    M: 'static + Clone + Sync + Send,
    A: 'static + SteppingAlg<M, R> + Send + Sync + Clone,
    R: 'static + SeedableRng + Rng + std::fmt::Debug,
{
    let mut rng = SeedableRng::from_rng(
        rng.write()
        .expect("Failed to get write access to rng")
        .deref_mut()
    ).expect("Failed to create seedable rng from input rng.");


    let mut stepper = stepper.clone();
    // let prior_sample = stepper.prior_sample(&mut rng, init_model);
    let prior_sample = init;

    //TODO - Randomly initialize all model values

    // WarmUp
    stepper.set_adapt(AdaptationMode::Enabled);

    let mut warmup_draws = if keep_warmup {
        (0..n_warmup)
            .scan(prior_sample.clone(), |m, _| { 
                *m = stepper.step(&mut rng, (*m).clone());
                Some(m.clone())
            }).collect()
   
    } else {
        let mp = (0..n_warmup)
            .fold(prior_sample.clone(), |m, _| {
                stepper.step(&mut rng, m)
            });
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
            *m = stepper.step(&mut rng, (*m).clone());
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
    extern crate test;
    use super::*;
    use rand::SeedableRng;
    use std::sync::{Arc, RwLock};
    use steppers::Mock;
    const SEED: [u8; 32] = [0; 32];

    #[test]
    fn draw_from_stepper_returns_sequence_with_mock_sampler() {
        let init: i32 = 0;
        let update = |x: i32| x + 1;
        let alg_start = Mock::new(init, update);
        let mut rng = rand::rngs::StdRng::from_seed(SEED);

        let results = draw_from_stepper(
            Arc::new(RwLock::new(&mut rng)),
            alg_start,
            init,
            10,
            10,
            1,
            true
        );

        assert_eq!(results.len(), 20);
        let expected: Vec<i32> = (1..21).collect();
        assert_eq!(results, expected);
    }

}
