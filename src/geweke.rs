//! Geweke Test Utils

use rv::misc::*;
use crate::stepper_traits::*;
use crate::utils::test::write_vec_file;
use std::env::temp_dir;

use rand::Rng;

/// Configuration for Geweke Tests
pub struct GewekeConfig {
    /// Number of steps to drop between draws
    pub thinning: usize,
    /// Number of samples to take
    pub n_samples: usize,
    /// Number of initial draws to drop
    pub burn_in: usize,
    /// Minimum cutoff for p-values from KS-test
    pub alpha: f64,
    /// Write debugging output
    pub save: bool,
}

impl GewekeConfig {
    /// Create a new Geweke Config
    pub fn new(n_samples: usize, burn_in: usize, thinning: usize, alpha: f64, save: bool) -> Self {
        Self {
            n_samples,
            thinning,
            burn_in,
            alpha,
            save,
        }
    }
}

fn resample_params<'a, Model, RNG>(
    stepper: &mut Box<dyn SteppingAlg<'a, Model, RNG> + 'a>,
    rng: &mut RNG,
    model: Model
) -> Model
where
    Model: Clone,
    RNG: Rng,
{
    let step = stepper.step_with_log_likelihood(rng, model, None);
    step.0
}

fn marginal_conditional_simulator<'a, Model, RNG, B>(
    builder: &B,
    init_model: Model,
    rng: &mut RNG,
    config: &GewekeConfig,
) -> Vec<Model> 
where
    Model: Clone,
    RNG: Rng,
    B: StepperBuilder<'a, Model, RNG>,
{
    let mut stepper = builder.build();
    stepper.adapt_enable();
    (0..config.burn_in).fold(init_model.clone(), |m, _| {
        stepper.step(rng, m)
    });
    stepper.adapt_disable();
    (0..(config.thinning * config.n_samples + config.burn_in)).map(|_| {
        let m = init_model.clone();
        let prior_draw = stepper.draw_prior(rng, m);
        resample_params(&mut stepper, rng, prior_draw)
    })
        .skip(config.burn_in)
        .step_by(config.thinning)
        .collect()
}

/// Successive Conditional Sampler
fn successive_conditional_sampler<'a, Model, RNG, B, RSF>(
    builder: &B,
    init_model: Model,
    resample_data: RSF,
    rng: &mut RNG,
    config: &GewekeConfig,
) -> Vec<Model>
where
    Model: Clone,
    RNG: Rng,
    B: StepperBuilder<'a, Model, RNG>,
    RSF: Fn(Model, &mut RNG) -> Model,
{
    let mut stepper = builder.build();
    stepper.adapt_disable();
    let prior_draw = stepper.draw_prior(rng, init_model);
    let init_state = resample_params(&mut stepper, rng, prior_draw);
    stepper.adapt_disable();

    (0..(config.thinning * config.n_samples + config.burn_in)).scan(init_state, |state, _| {
        let m = resample_params(&mut stepper, rng, state.clone());
        *state = resample_data(m, rng);
        Some(state.clone())
    })
        .skip(config.burn_in)
        .step_by(config.thinning)
        .collect()
}

/// Preform a geweke test for sampler correctness
pub fn geweke_test<'a, Model, RNG, B, SF, RSF>(
    config: GewekeConfig,
    stepper_builder: B,
    init_model: Model,
    to_stats: SF,
    resample_data: RSF,
    rng: &mut RNG,
) -> bool
where
    Model: Clone,
    RNG: Rng,
    B: StepperBuilder<'a, Model, RNG>,
    SF: Fn(&Model) -> Vec<f64>,
    RSF: Fn(Model, &mut RNG) -> Model,
{

    let mcs: Vec<Vec<f64>> = marginal_conditional_simulator(
        &stepper_builder,
        init_model.clone(),
        rng,
        &config
    )
        .iter()
        .map(&to_stats)
        .collect();

    let scs: Vec<Vec<f64>> = successive_conditional_sampler(
        &stepper_builder,
        init_model,
        resample_data,
        rng,
        &config
    )
        .iter()
        .map(&to_stats)
        .collect();
    
    let mut result = true;
    for stat_num in 0..mcs[0].len() {
        let a: Vec<f64> = mcs.iter().map(|m| m[stat_num]).collect();
        let b: Vec<f64> = scs.iter().map(|m| m[stat_num]).collect();

        let (_, p) = ks_two_sample(&a, &b, KsMode::Auto, KsAlternative::TwoSided)
            .expect("Failed to get result from ks_two_sample");
        if config.save {
            let tmp = temp_dir();
            let mut mcs_path = tmp.clone();
            mcs_path.push(format!("/tmp/{}_mcs.txt", stat_num));
            let mut scs_path = tmp.clone();
            scs_path.push(format!("/tmp/{}_scs.txt", stat_num));

            eprintln!("mcs_path = {:?}", mcs_path);
            eprintln!("scs_path = {:?}", scs_path);

            write_vec_file(&a, mcs_path.as_path()).unwrap();
            write_vec_file(&b, scs_path.as_path()).unwrap();
        }
        result &= p > config.alpha;
    }

    result
}
