use rand::rngs::StdRng;
use rand::SeedableRng;
use rmcmc::steppers::srwm::SRWMBuilder;
use rmcmc::utils::log_likelihood_from_data;
use rmcmc::{make_lens, Lens, Parameter, Runner};
use rv::dist::{Gamma, Poisson};
use rv::misc::ks_test;
use rv::traits::Cdf;

#[test]
pub fn poisson_1d() {
    #[derive(Clone)]
    struct Model {
        mean: f64,
    }

    impl Default for Model {
        fn default() -> Self {
            Model { mean: 0.0 }
        }
    }

    let data: Vec<u32> = vec![2, 3, 2, 1, 1, 3, 2, 4, 3, 4];

    let log_likelihood =
        log_likelihood_from_data(&data, |m: &Model| Poisson::new(m.mean));

    let parameter = Parameter::new(
        Gamma::new(3.0, 3.0).unwrap(),
        make_lens!(Model, f64, mean),
    );

    let builder = SRWMBuilder::new(&parameter, &log_likelihood, 10.0, 1.0);

    let runner = Runner::new(&builder).chains(1).warmup(1000).draws(1000);

    let mut rng = StdRng::seed_from_u64(0xFEED);
    let sample: Vec<Vec<Model>> = runner
        .thinning(10)
        .warmup(1000)
        .draws(2000)
        .run(&mut rng);

    let rates: Vec<f64> = sample
        .iter()
        .flatten()
        .map(|m| m.mean)
        .collect();

    let data_sum: u32 = data.iter().sum();

    let posterior =
        Gamma::new(3.0 + data_sum as f64, 3.0 + data.len() as f64)
            .unwrap();

    let (stat, pvalue) = ks_test(&rates, |x| posterior.cdf(&x));
    println!("stat = {}, pvalue = {}", stat, pvalue);
    assert!(pvalue > 0.05)
}
