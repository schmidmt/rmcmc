//! Stochastic volatility of the daily SP 500
//!
//! ν ~ exp(0.1)
//! σ ~ exp(50)
//! s_i ~ N(s_{i-1}, σ^2)
//! log(r_i) ~ t(ν, 0, exp(-2 s_i))

use std::io;
use std::fs::File;
use std::f64::consts::PI;

use env_logger;
use log::debug;
use csv;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rv::dist::{Gaussian, Exponential};
use rv::prelude::*;
use special::Gamma as SGamma;
use nalgebra::{DVector, DMatrix};

use rmcmc::*;
use rmcmc::steppers::srwm::SRWMBuilder;
use rmcmc::steppers::group::GroupBuilder;

#[derive(Debug, Clone)]
struct GaussianRW {
    inner: Gaussian,
    len: usize,
}

#[derive(Debug, Clone)]
enum Error {
    Failure
}

impl GaussianRW {
    pub fn new(mu: f64, sigma: f64, len: usize) -> Result<Self, Error> {
        let inner = match Gaussian::new(mu, sigma) {
            Ok(x) => x,
            Err(_) => return Err(Error::Failure),
        };
        Ok(Self {
            inner,
            len
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn mu(&self) -> f64 {
        self.inner.mu()
    }

    pub fn sigma(&self) -> f64 {
        self.inner.sigma()
    }
}

impl Rv<DVector<f64>> for GaussianRW {
    fn draw<R: Rng>(&self, rng: &mut R) -> DVector<f64> {
        let start = self.inner.draw(rng);
        let it = (0..self.len).scan(start, |state, _| {
            let innov: f64 = self.inner.draw(rng);
            *state = *state + innov;
            Some(*state)
        });
        DVector::from_iterator(self.len, it)
    }

    fn ln_f(&self, x: &DVector<f64>) -> f64 {
        let pairs = x
            .iter()
            .take(self.len - 1)
            .zip(x.iter().skip(1));

        pairs.fold(0.0, |acc, (p, c)| {
            let g = Gaussian::new_unchecked(self.mu() + p, self.sigma());
            acc + g.ln_f(c)
        })
    }
}

fn main() -> io::Result<()> {
    env_logger::init();
    let mut rng = Xoshiro256Plus::seed_from_u64(0);

    let file = File::open("./examples/sp500.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    let deltas: Vec<f64> = rdr.records().map(|r| {
        let record = r.unwrap();
        record.get(1).unwrap().parse::<f64>().unwrap()
    })
    .collect();

    let n_deltas: usize = deltas.len();

    #[derive(Debug, Clone)]
    struct Model {
        nu: f64,
        sigma: f64,
        q: DVector<f64>,
    }

    impl Default for Model {
        fn default() -> Self {
            Self {
                nu: 1.0,
                sigma: 1.0,
                q: DVector::zeros(0),
            }
        }
    }

    impl Model {
        fn from_deltas(deltas: &Vec<f64>) -> Self {
            Self {
                nu: 1.0,
                sigma: 1.0,
                q: DVector::zeros(deltas.len()),
            }
        }
    }

    let nu_parameter = Parameter::new_independent(
        Exponential::new_unchecked(0.1),
        make_lens!(Model, nu),
    );

    let sigma_parameter = Parameter::new_independent(
        Exponential::new_unchecked(50.0),
        make_lens!(Model, sigma),
    );

    debug!("sigma prior for ln_f(-1) = {}", sigma_parameter.prior(&Model::default()).ln_f(&-1.0));

    let q_parameter = Parameter::new_dependent(
        Box::new(move |s: &Model| {
            debug!("sigma = {}", s.sigma);
            GaussianRW::new(0.0, s.sigma, n_deltas).unwrap()
        }),
        make_lens!(Model, q),
    );

    let log_likelihood = |m: &Model| -> f64 {
        //println!("model = {:?}", m);
        let ll = deltas.iter().zip(m.q.iter()).map(|(x, s)| {
            let vp1 = (m.nu + 1.0) / 2.0;
            let xterm = -vp1 * (1.0_f64 + (-2.0 * s).exp() * f64::from(*x).powi(2) / m.nu).ln();
            let zterm = vp1.ln_gamma().0
                - (m.nu / 2.0).ln_gamma().0
                - 0.5 * (m.nu * PI).ln()
                - s;
            zterm + xterm 
        }).sum();
        debug!("LL = {}", ll);
        ll
    };

    let nu_stepper = SRWMBuilder::new(
        &nu_parameter,
        &log_likelihood,
        3.0,
        1.0
    );

    let sigma_stepper = SRWMBuilder::new(
        &sigma_parameter,
        &log_likelihood,
        3.0,
        1.0
    );

    let s_stepper = SRWMBuilder::new(
        &q_parameter,
        &log_likelihood,
        DVector::zeros(n_deltas),
        DMatrix::identity(n_deltas, n_deltas),
    );

    let group_builder = GroupBuilder::new(vec![
        &nu_stepper,
        &sigma_stepper,
        &s_stepper,
    ]);

    let init = Model::from_deltas(&deltas);

    let samples: Vec<Vec<Model>> = Runner::new(&group_builder)
        .warmup(10)
        .thinning(1)
        .chains(1)
        .draws(100)
        .initial_model(init)
        .run(&mut rng);

    samples.iter().flatten().for_each(|m| {
        println!("{}, {}, {:?}", m.nu, m.sigma, m.q);
    });

    Ok(())
}
