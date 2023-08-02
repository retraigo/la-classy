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

pub fn minibatch_stochastic_gradient_descent(
    config: &ModelConfig,
    data: &DMatrix<f64>,
    targets: &DVector<f64>,
) -> DVector<f64> {
    let mut eta = config.learning_rate;
    let mut rng = rand::thread_rng();
    let mut weights = DVector::from_element(data.ncols(), 1.0);
    let mut optimizer = match config.optimizer {
        Optimizer::Adam(config) => Opt::Adam(AdamOptimizer::new(config, weights.len())),
        Optimizer::None => Opt::NoOptimizer(NoOptimizer::new()),
    };
    for epoch in 0..config.epochs {
        let batch_size = data.nrows() / config.n_batches;
        for _ in 0..(config.n_batches) {
            let j = rng.gen_range(0..config.n_batches);
            let remaining = data.nrows() - (j * batch_size);
            let current_batch_size = if remaining < batch_size {
                remaining
            } else {
                batch_size
            };
            let inverse_batch_size = 1.0 / current_batch_size as f64;
            let batch_data = data.rows(j * batch_size, current_batch_size);
            let mut h = batch_data * &weights;
            match config.model {
                Model::None => (),
                Model::Logit => h.apply(|x| {
                    let res = sigmoid(*x);
                    *x = res;
                }),
            };
            let errors = &h - &targets.rows(j * batch_size, current_batch_size);
            eta = get_learning_rate(&config.scheduler, eta, epoch, config.learning_rate);
            let gradient = &batch_data.transpose() * &errors * inverse_batch_size;
            let l1 = config.c * &weights.map(|w| if w >= 0.0 { 1.0 } else { -1.0 });

            optimizer.optimize(&mut weights, gradient, eta, l1);
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
