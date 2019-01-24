use steppers::SteppingAlg;
use rand::Rng;

/// statistics monitoring via a summarizer
pub trait Summarizer<A, M, R: Rng> {
    type Output;
    type S: SteppingAlg<M, R>;
    fn on_step(prev: A, steppers: &[Box<Self::S>]) -> A;
    fn finalize(state: A) -> Self::Output;
}

struct NullSummary();
struct DefaultSummarizer();


/*
impl Summarizer<NullSummary> for DefaultSummarizer 
{

}
*/
