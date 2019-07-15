use rv::dist::Gaussian;
use rv::prelude::*;
use rv::misc::ks_test;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rmcmc::{Parameter, Runner};
use rmcmc::make_lens;
use rmcmc::Lens;
use rmcmc::steppers::srwm::SRWMBuilder;

#[test]
fn mean_of_gaussians() {
    let mut rng = StdRng::seed_from_u64(0xFEED_u64);
    let n_data = 10;
    let rv = Gaussian::new(1.0, 1.0).unwrap();

    let data: Vec<f64> = rv.sample(n_data, &mut rng);

    println!("{:?}", data);

    let prior = Gaussian::new(0.0, 1.0).unwrap();

    let data_sum: f64 = data.iter().sum();
    let posterior_mu = 1.0 / (1.0 + (n_data as f64)) * data_sum;
    let posterior_sigma = 1.0 / (1.0 + (n_data as f64));
    let posterior = Gaussian::new(posterior_mu, posterior_sigma.sqrt()).unwrap();

    #[derive(Clone, Debug)]
    struct Model {
        mean: f64
    }

    impl Default for Model {
        fn default() -> Self {
            Self {
                mean: 0.0
            }
        }
    }

    let parameter = Parameter::new(
        prior,
        make_lens!(Model, f64, mean)
    );

    let log_likelihood = |m: &Model| {
        let g = Gaussian::new(m.mean, 1.0).unwrap();
        data.iter().map(|d| { g.ln_f(d) }).sum()
    };


    let stepper_builder = SRWMBuilder::new(&parameter, &log_likelihood);


    let runner = Runner::new(&stepper_builder)
        .draws(1000)
        .warmup(1000)
        .thinning(10)
        .chains(1);

    let sample: Vec<f64> = runner.run(&mut rng)
        .iter()
        .flatten()
        .map(|m| m.mean)
        .collect();


    let (stat, p) = ks_test(&sample, |x| {posterior.cdf(&x)});

    println!("stat = {}, p = {}", stat, p);
    println!("values = {:?}", sample);
    assert!(p < 0.05);
}