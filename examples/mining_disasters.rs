use std::io;

extern crate rmcmc;
extern crate rv;
extern crate rand;

use rmcmc::*;
use rv::prelude::Rv;


fn main() -> io::Result<()> {
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(0);

    const N_DATA: usize = 111;
    const DISASTER_DATA: [u32; N_DATA] = [
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
        3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
        2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
        0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1];

    #[derive(Clone, Debug)]
    struct Model {
        pub switch_point: u32,
        pub early_mean: f64,
        pub late_mean: f64,
    }

    let switch_point_parameter = Parameter::new(
        "switch_point".to_owned(),
        rv::dist::DiscreteUniform::new(0, N_DATA as u32).unwrap(),
        make_lens!(Model, u32, switch_point)
    );

    let early_mean_parameter = Parameter::new(
        "early_mean".to_owned(),
        rv::dist::InvGamma::new(3.0, 0.1).unwrap(),
        make_lens!(Model, f64, early_mean)
    );

    let late_mean_parameter = Parameter::new(
        "late_mean".to_owned(),
        rv::dist::InvGamma::new(3.0, 0.1).unwrap(),
        make_lens!(Model, f64, late_mean)
    );

    fn loglikelihood(m: &Model) -> f64 {
        let early_dist_r = rv::dist::Poisson::new(m.early_mean);
        let late_dist_r = rv::dist::Poisson::new(m.late_mean);
        
        let res = early_dist_r.and_then(|early_dist| {
            late_dist_r.map(|late_dist| {
                DISASTER_DATA.iter().enumerate().fold(0.0, |sum, (i, x)| {
                    if i < m.switch_point as usize {
                        sum + early_dist.ln_f(x)
                    } else {
                        sum + late_dist.ln_f(x)
                    }
                })
            })
        });
        res.unwrap_or(std::f64::NEG_INFINITY)
    };

    let sub_steppers: Vec<Box<steppers::SteppingAlg<Model, _>>> = vec![
        Box::new(steppers::SRWM::new(early_mean_parameter, loglikelihood, None).unwrap()),
        Box::new(steppers::SRWM::new(late_mean_parameter, loglikelihood, None).unwrap()),
        Box::new(steppers::DiscreteSRWM::new(switch_point_parameter, loglikelihood, None).unwrap())
    ];
    
    let stepper = steppers::Group::new(sub_steppers);

    let init = Model {
        switch_point: 10,
        early_mean: 7.0,
        late_mean: 5.0,
    };

    let results = Runner::new(stepper)
        .warmup(5000)
        .samples(2000)
        .chains(4)
        .thinning(50)
        .run(&mut rng, init);

    results.iter().flatten().for_each(|m| println!("{:?}", m));

    println!("early_mean rHat = {}", diagnostics::rhat(results.iter().map(|c| c.iter().map(|m| m.early_mean).collect::<Vec<f64>>()).collect()));
    println!("late_mean rHat = {}", diagnostics::rhat(results.iter().map(|c| c.iter().map(|m| m.late_mean).collect::<Vec<f64>>()).collect()));
    println!("switch_point rHat = {}", diagnostics::rhat(results.iter().map(|c| c.iter().map(|m| f64::from(m.switch_point)).collect::<Vec<f64>>()).collect()));


    Ok(())
}
