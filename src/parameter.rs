use rand::Rng;
use rv::traits::Rv;

use crate::lens::*;

/// Parameter Struct
/// D: Rv Implementation
/// T: Parameter Type
/// S: State Type
pub struct Parameter<R, T, S>
where
    R: Rv<T>,
{
    /// Prior distribution
    pub prior: R,
    /// Lens to update value
    pub lens: Lens<T, S>,
}

impl<D, T, S> Clone for Parameter<D, T, S>
where
    D: Rv<T> + Clone,
{
    fn clone(&self) -> Parameter<D, T, S> {
        Parameter {
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
    pub fn new(prior: D, lens: Lens<T, S>) -> Self {
        Parameter { prior, lens }
    }

    /// Create a new version of the struct with a randomly drawn value from the value's
    /// distribution.
    pub fn draw<R: Rng>(&self, s: &S, rng: &mut R) -> S {
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

        let _ = Parameter::new(Beta::jeffreys(), make_lens!(Foo, f64, bar));
    }
}
