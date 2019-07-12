use num_traits::Float;

/// Wrapper for Mean and Variance
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct MeanAndVariance<T: Float> {
    /// Mean of given data
    pub mean: T,
    /// Variance of given data
    pub variance: T,
    /// Number of data encountered
    pub count: usize,
}

impl<T: Float> Default for MeanAndVariance<T> {
    fn default() -> Self {
        Self {
            mean: T::from(0.0).unwrap(),
            variance: T::from(0.0).unwrap(),
            count: 0,
        }
    }
}

impl<T: Float> MeanAndVariance<T> {
    /// Create a new Mean and Variance wrapper
    pub fn new(mean: T, variance: T, count: usize) -> Self {
        Self {
            mean,
            variance,
            count,
        }
    }

    /// Create a new MeanAndVariance with updated values
    pub fn update(&self, values: &[T]) -> Self {
        values.iter().fold(*self, |acc, &x| {
            let count = acc.count + 1;
            let delta = T::from(x).unwrap() - acc.mean;
            let mean = acc.mean + delta / T::from(count).unwrap();
            let delta2 = T::from(x).unwrap() - mean;
            let variance = delta * delta2;
            Self {
                mean,
                variance,
                count,
            }
        })
    }

    /// Calculate mean and variance for given slice `values`
    pub fn from_values(values: &[T]) -> Self {
        MeanAndVariance::default().update(values)
    }

    /// Determine the standard deviation
    pub fn std(&self) -> T {
        self.variance.sqrt()
    }
}
