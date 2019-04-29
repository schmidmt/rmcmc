use std::error::Error;
use std::fmt::Display;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::Path;
use rv::traits::*;
use std::marker::PhantomData;
use rand::Rng;
use num::Float;

#[cfg(test)]
pub mod geweke;

pub fn multiple_tries<F: FnMut(usize) -> bool>(
    n_tries: usize,
    mut f: F,
) -> bool {
    for i in 0..n_tries {
        println!("MULTIPLE_TRIES: {}", i);
        if f(i) {
            return true;
        }
    }
    false
}

pub fn write_samples_to_file<T: Display>(
    path: &Path,
    samples: &[T],
) -> io::Result<()> {
    let display = (*path).display();
    let mut file = match File::create(&path) {
        Err(why) => {
            panic!("couldn't create {}: {}", display, why.description())
        }
        Ok(file) => file,
    };

    let string_samples: Vec<String> =
        samples.iter().map(|x| format!("{}", x)).collect();
    let output = string_samples.join("\n");
    match file.write_all(output.as_bytes()) {
        Err(why) => {
            panic!("couldn't write to {}: {}", display, why.description())
        }
        Ok(_) => Ok(()),
    }
}

#[derive(Clone, Debug)]
pub struct MultiRv<X, T>
where
    T: Rv<X>
{
    dims: usize,
    base: T,
    phantom_x: PhantomData<X>
}

impl<X, T> MultiRv<X, T>
where
    T: Rv<X>
{
    pub fn new(dims: usize, base: T) -> Self {
        Self {
            dims,
            base,
            phantom_x: PhantomData
        }
    }
}

impl<X, T> Rv<Vec<X>> for MultiRv<X, T>
where
    T: Rv<X>
{
    fn ln_f(&self, x: &Vec<X>) -> f64 {
        x.iter().fold(0.0, |a, y| self.base.ln_f(y) + a)
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> Vec<X> {
        self.base.sample(self.dims, rng)
    }
}

impl<X, T> Support<Vec<X>> for MultiRv<X, T>
where
    T: Rv<X> + Support<X>
{
    fn supports(&self, x: &Vec<X>) -> bool {
        x.iter().all(|y| self.base.supports(y))
    }
}

impl<X, T> ContinuousDistr<Vec<X>> for MultiRv<X, T>
where
    T: Rv<X> + Support<X>
{}

impl<X, T> DiscreteDistr<Vec<X>> for MultiRv<X, T>
where
    T: Rv<X> + Support<X>
{}

impl<X, T> Mean<Vec<X>> for MultiRv<X, T>
where
    T: Rv<X> + Mean<X>,
    X: Clone
{
    fn mean(&self) -> Option<Vec<X>> {
        self.base.mean().map(|m| (0..self.dims).map(|_| m.clone()).collect())
    }
}

impl<X, T> Median<Vec<X>> for MultiRv<X, T>
where
    T: Rv<X> + Median<X>,
    X: Clone
{
    fn median(&self) -> Option<Vec<X>> {
        self.base.median().map(|m| (0..self.dims).map(|_| m.clone()).collect())
    }
}

impl<X, T> Mode<Vec<X>> for MultiRv<X, T>
where
    T: Rv<X> + Mode<X>,
    X: Clone
{
    fn mode(&self) -> Option<Vec<X>> {
        self.base.mode().map(|m| (0..self.dims).map(|_| m.clone()).collect())
    }
}

impl<X, T> Variance<Vec<X>> for MultiRv<X, T>
where
    T: Rv<X> + Variance<X>,
    X: Clone
{
    fn variance(&self) -> Option<Vec<X>> {
        self.base.variance().map(|m| (0..self.dims).map(|_| m.clone()).collect())
    }
}


#[derive(Clone, Copy, Debug)]
pub struct MeanAndVariance<T: Float> {
    pub mean: T,
    pub variance: T,
    pub count: usize,
}

impl<T: Float> MeanAndVariance<T> {
    pub fn new(mean: T, variance: T, count: usize) -> Self {
        Self { mean, variance, count }
    }

    pub fn empty() -> Self {
        Self { mean: T::from(0.0).unwrap(), variance: T::from(0.0).unwrap(), count: 0 }
    }

    pub fn from_values<F: Float>(vals: &[F]) -> Self {
        vals.iter().fold(MeanAndVariance::empty(), |acc, &x| {
            let count = acc.count + 1;
            let delta = T::from(x).unwrap() - acc.mean;
            let mean = acc.mean + delta / T::from(count).unwrap();
            let delta2 = T::from(x).unwrap() - mean;
            let variance = delta * delta2;
            Self { mean, variance, count }
        })
    }

    pub fn std(&self) -> T {
        self.variance.sqrt()
    }
}
