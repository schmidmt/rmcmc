//! Benchmark the SRWM implementation for vectors

use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DVector, DMatrix};
use rv::prelude::*;
use rand_xoshiro::Xoshiro256Plus;
use rand::SeedableRng;

use rmcmc::*;
use rmcmc::steppers::srwm::SRWMBuilder;

#[derive(Clone, Debug)]
struct Model {
    inner: DVector<f64>,
}

impl Default for Model {
    fn default() -> Self {
        Self { inner: DVector::zeros(0) }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = Xoshiro256Plus::seed_from_u64(0);

    let mut sample = |dims: usize, adapt: bool| {
        let inner_dist = Gaussian::new_unchecked(0.0, 1.0);
        let sa = DMatrix::from_fn(dims, dims, |_, _| inner_dist.draw(&mut rng));
        let sa = &sa * &sa.transpose();
        let mu: DVector<f64> = DVector::from_fn(dims, |_, _| inner_dist.draw(&mut rng));
        let data = MvGaussian::new(mu, sa.clone()).unwrap().sample(10, &mut rng);

        let log_likelihood = |m: &Model| {
            let dist = MvGaussian::new(m.inner.clone(), sa.clone()).unwrap();
            data.iter().fold(0.0, |acc, x| {
                acc + dist.ln_f(x)
            })
        };

        let parameter = Parameter::new_independent(
            MvGaussian::new_unchecked(DVector::zeros(dims), DMatrix::identity(dims, dims)),
            make_lens!(Model, inner)
        );

        let mut stepper = SRWMBuilder::new(
            &parameter,
            &log_likelihood,
            DVector::zeros(dims),
            DMatrix::identity(dims, dims),
        )
            .build();

        if adapt {
            stepper.adapt_enable();
        } else {
            stepper.adapt_disable();
        }

        let model = Model {
            inner: DVector::zeros(dims),
        };

        (0..100).fold(model, |m, _| stepper.step(&mut rng, m));
    };

    let mut group = c.benchmark_group("dimensions");

    for dims in 1..10 {
        group.bench_with_input(format!("100 Steps with {} dims (non-adapting)", dims), &dims, |b, &dims| {
            b.iter(|| sample(dims, false))
        });
        group.bench_with_input(format!("100 Steps with {} dims (adapting)", dims), &dims, |b, &dims| {
            b.iter(|| sample(dims, true))
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
