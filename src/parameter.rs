use std::sync::Arc;

use rand::Rng;
use rv::traits::Rv;

use crate::lens::*;

/// An Rv generator
pub type RvFunc<R, S> = dyn Fn(&S) -> R + Send + Sync;

/// Parameter which will be updated by a stepper
pub enum Parameter<R, T, S>
    where
        R: Rv<T>,
{
    /// A parameter which depends on another value
    Dependent {
        /// Generator from state to Rv (prior)
        generator: Box<RvFunc<R, S>>,
        /// Lens to access value from the state
        lens: Lens<T, S>,
    },
    /// A parameter which is independent of other parameters
    Independent {
        /// Prior distribution
        dist: Arc<R>,
        /// Lens to access value from the state
        lens: Lens<T, S>,
    },
}

impl<R, T, S> Parameter<R, T, S>
    where
        R: Rv<T>,
{
    /// Create a new independent parameter
    pub fn new_independent(prior: R, lens: Lens<T, S>) -> Self {
        Parameter::Independent {
            dist: Arc::new(prior),
            lens,
        }
    }

    /// Create a new dependent parameter
    pub fn new_dependent(generator: Box<RvFunc<R, S>>, lens: Lens<T, S>) -> Self {
        Parameter::Dependent {
            generator,
            lens,
        }
    }

    /// Create a new version of the struct with a randomly drawn value from the value's
    /// distribution.
    pub fn draw<RN: Rng>(&self, s: &S, rng: &mut RN) -> S {
        let new_value = self.prior(s).draw(rng);
        self.lens().set(s, new_value)
    }

    /// Retreive the parameters lens
    pub fn lens(&self) -> &Lens<T, S> {
        match self {
            Parameter::Independent {dist: _ , lens} => &lens,
            Parameter::Dependent {generator: _ , lens} => &lens,
        }
    }

    /// Retreive the prior
    pub fn prior(&self, state: &S) -> Arc<R> {
        match self {
            Parameter::Independent { dist , lens: _ } => Arc::clone(dist),
            Parameter::Dependent { generator, lens: _ } => {
                let gen = generator(state);
                Arc::new(gen)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rv::dist::{Beta, Geometric, MvGaussian};
    use nalgebra::DVector;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn new_independent() {
        struct Foo {
            bar: f64,
        }

        let _ = Parameter::new_independent(Beta::jeffreys(), make_lens!(Foo, f64, bar));
    }

    #[test]
    fn new_dependent() {
        let mut rng = StdRng::seed_from_u64(0xABCD);

        #[derive(Debug, PartialEq)]
        struct Foo {
            x: f64,
            y: u32,
        }

        let state = Foo { x: 0.0, y: 5 };

        let param_x = Parameter::new_independent(Beta::jeffreys(), make_lens!(Foo, f64, x));
        let param_y = Parameter::new_dependent(Box::new(|s: &Foo| {
            Geometric::new_unchecked(s.x)
        }), make_lens!(Foo, u32, y));

        let state = param_x.draw(&state, &mut rng);
        let state = param_y.draw(&state, &mut rng);

        assert_eq!(
            state,
            Foo { x: 0.05698665526084659, y: 3 },
        );
    }

    #[test]
    fn vec_param() {
        let mut rng = StdRng::seed_from_u64(0xABCD);

        struct Foo {
            xs: DVector<f64>,
        }
    
        let param = Parameter::new_independent(
            MvGaussian::standard(2).unwrap(),
            make_lens!(Foo, xs)
        );
        let state = Foo { xs: DVector::zeros(2) };
        let state = param.draw(&state, &mut rng);

        assert!(state.xs.relative_eq(
            &DVector::from_column_slice(&[
                -0.6497147171858277,
                -1.2214173483049675
            ]),
            1E-10,
            1E-10
        ));
    }
}
