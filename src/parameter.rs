use rand::RngCore;
use rv::traits::Rv;
use std::fmt;

use crate::lens::*;

/// Parameter Struct
/// D: Rv Implementation
/// T: Parameter Type
/// S: State Type
pub struct Parameter<R, T, S>
where
    R: Rv<T>,
{
    /// Name of parameter (must be unique)
    pub name: String,
    /// Prior distribution
    pub prior: R,
    /// Lens to update value
    pub lens: Lens<T, S>,
}

impl<D, T, S> fmt::Debug for Parameter<D, T, S>
where
    D: Rv<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parameter {{ name: {} }}", self.name)
    }
}

impl<D, T, S> Clone for Parameter<D, T, S>
where
    D: Rv<T> + Clone,
{
    fn clone(&self) -> Parameter<D, T, S> {
        Parameter {
            name: self.name.clone(),
            prior: self.prior.clone(),
            lens: self.lens.clone(),
        }
    }
}

/// Parameter Mapping
impl<D, T, S> Parameter<D, T, S>
where
    D: Rv<T>,
{
    /// Create a new parameter container
    /// # Arguments
    /// * `name` - Name of the parameter for output labeling
    /// * `prior` - Prior probability of random variable
    /// * `lens` - Lens to access inner value
    pub fn new(name: String, prior: D, lens: Lens<T, S>) -> Self {
        Parameter { name, prior, lens }
    }

    /// Create a new version of the struct with a randomly drawn value from the value's
    /// distribution.
    pub fn draw<R: RngCore>(&self, s: &S, rng: &mut R) -> S {
        let new_value = self.prior.draw(rng);
        self.lens.set(s, new_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rv::dist::Beta;

    #[test]
    fn new() {
        struct Foo {
            bar: f64,
        }

        let p = Parameter::new(
            "test".to_string(),
            Beta::jeffreys(),
            make_lens!(Foo, f64, bar),
        );
        assert_eq!(p.name, "test".to_string());
    }
}
