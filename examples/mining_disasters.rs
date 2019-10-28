use rmcmc::*;
use rv::prelude::Rv;
use std::io;
use rmcmc::steppers::discrete_srwm::DiscreteSRWMBuilder;
use rmcmc::steppers::srwm::SRWMBuilder;
use rmcmc::steppers::group::GroupBuilder;


fn main() -> io::Result<()> {
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(0);

    const N_DATA: usize = 111;
    const DISASTER_DATA: [u32; N_DATA] = [
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4,
        4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2,
        1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
    ];

    #[derive(Clone, Debug)]
    struct Model {
        pub switch_point: u32,
        pub early_mean: f64,
        pub late_mean: f64,
    }

    impl Default for Model {
        fn default() -> Self {
            unimplemented!()
        }
    }

    let switch_point_parameter = Parameter::new_independent(
        rv::dist::DiscreteUniform::new(0, N_DATA as u32).unwrap(),
        make_lens!(Model, u32, switch_point),
    );

    let early_mean_parameter = Parameter::new_independent(
        rv::dist::InvGamma::new(3.0, 0.1).unwrap(),
        make_lens!(Model, f64, early_mean),
    );

    let late_mean_parameter = Parameter::new_independent(
        rv::dist::InvGamma::new(3.0, 0.1).unwrap(),
        make_lens!(Model, f64, late_mean),
    );

    fn log_likelihood(m: &Model) -> f64 {
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

    let switch_point_stepper = DiscreteSRWMBuilder::new(
        &switch_point_parameter,
        &log_likelihood
    );

    let early_mean_stepper = SRWMBuilder::new(
        &early_mean_parameter,
        &log_likelihood,
        3.0,
        1.0
    );

    let late_mean_stepper = SRWMBuilder::new(
        &late_mean_parameter,
        &log_likelihood,
        3.0,
        1.0
    );

    let group_builder = GroupBuilder::new(vec![
        &early_mean_stepper, 
        &switch_point_stepper,
        &late_mean_stepper
    ]);

    let init = Model {
        switch_point: 10,
        early_mean: 7.0,
        late_mean: 5.0,
    };

    let samples: Vec<Vec<Model>> = Runner::new(&group_builder)
        .warmup(1000)
        .thinning(20)
        .chains(4)
        .draws(2000)
        .initial_model(init)
        .run(&mut rng);

    samples.iter().flatten().for_each(|m| {
        println!("{},{},{}", m.early_mean, m.switch_point, m.late_mean);
    });
    Ok(())
}
