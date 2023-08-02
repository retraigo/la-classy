use nalgebra::{DMatrix, DVector};
use rand::Rng;

use crate::{
    core::{
        functions::sigmoid,
        loss::{binary_cross_entropy, cross_entropy, mean_absolute_error, mean_squared_error},
        optimizers::{adam::AdamOptimizer, noop::NoOptimizer, Optimizer as Opt},
        scheduler::get_learning_rate,
    },
    types::{LossFunction, Model, ModelConfig, Optimizer},
};

pub fn stochastic_gradient_descent(
    config: &ModelConfig,
    data: &DMatrix<f64>,
    targets: &DVector<f64>,
) -> DVector<f64> {
    let mut eta = config.learning_rate;
    let inverse_n = 1.0 / data.nrows() as f64;
    let mut rng = rand::thread_rng();
    let mut weights = DVector::from_element(data.ncols(), 1.0);

    let mut optimizer = match config.optimizer {
        Optimizer::Adam(config) => Opt::Adam(AdamOptimizer::new(config, weights.len())),
        Optimizer::None => Opt::NoOptimizer(NoOptimizer::new()),
    };
    for epoch in 0..config.epochs {
        for _ in 0..data.nrows() {
            let j = rng.gen_range(0..data.nrows());
            let current_data = data.row(j);

            let mut h = current_data * &weights;

            match config.model {
                Model::None => (),
                Model::Logit => h.apply(|x| {
                    let res = sigmoid(*x);
                    *x = res;
                }),
            };
            let errors = h.add_scalar(-*targets.get(j).unwrap());
            eta = get_learning_rate(&config.scheduler, eta, epoch, config.learning_rate);
            let gradient = &errors * &current_data * inverse_n;
            let l1 = config.c * &weights.map(|w| if w >= 0.0 { 1.0 } else { -1.0 });

            optimizer.optimize(&mut weights, gradient.transpose(), eta, l1);
        }
        if epoch % 100 == 0 && !config.silent {
            let mut h = data * &weights;

            match config.model {
                Model::None => (),
                Model::Logit => h.apply(|x| {
                    let res = sigmoid(*x);
                    *x = res;
                }),
            };
            let error: f64 = match config.loss {
                LossFunction::BinCrossEntropy => binary_cross_entropy(&targets, &h),
                LossFunction::CrossEntropy => cross_entropy(&targets, &h),
                LossFunction::MAE => mean_squared_error(&targets, &h),
                LossFunction::MSE => mean_absolute_error(&targets, &h),
            };
            println!("Epoch <{}: Current Errors {}", epoch, error);
        }
    }
    weights
}
