use steppers::SteppingAlg;

pub enum StatisticValue {
    AcceptanceRate(f64),
    LogLikelihood(f64)
}

pub struct Statistic<M, R> {
    steppers: Box<SteppingAlg<M, R>>,
    value: StatisticValue
}
