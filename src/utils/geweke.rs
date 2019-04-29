//! Implements the Geweke Joint Distribution Test
// # Geweke Joint Distribution Test
// More info can be found here:
// http://qed.econ.queensu.ca/pub/faculty/ferrall/quant/papers/04_04_29_geweke.pdf

/*
use steppers::SteppingAlg;
use rand::Rng;

trait GewekeJDTest<'a, M, R>: SteppingAlg<'a, M, R>
where
    M: 'a + Clone,
    R: 'a + Rng,
{
    fn loglikelihood(&self, model: &M) -> f64;
    fn resample_data(&self, model: M) -> M;

    fn marginal_conditional_simulator(&self, rng: &mut R, model: M, n_iterations: usize) -> Vec<M> {
        (0..n_iterations).map(|_| self.resample_data(self.prior_draw(rng, model.clone()))).collect()
    }

    /*
    fn successive_conditional_simulator(&mut self, rng: &mut R, model: M, n_iterations: usize) -> Vec<M> {
        (0..n_iterations).scan(self.resample_data(self.prior_draw(rng, model.clone())), |m, _| {
            self.
            *m = self.step(rng, m.clone());
            Some(m.clone())
        }).collect()
    }
    */

}

*/
