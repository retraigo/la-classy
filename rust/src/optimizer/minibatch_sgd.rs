extern crate nalgebra as na;
use crate::common::{
    functions::{binary_cross_entropy, mean_squared_error, sigmoid},
    types::{LossFunction, Model},
};
use na::{DMatrix, DVector};
use rand::Rng;

pub fn minibatch_stochastic_gradient_descent_optimizer(
    data: &DMatrix<f64>,
    target: &DVector<f64>,
    init_weights: &DVector<f64>,
    loss: LossFunction,
    model: Model,
    fit_intercept: bool,
    epochs: usize,
    silent: bool,
    learning_rate: f64,
    n_batches: usize,
) -> (DVector<f64>, f64) {
    let mut intercept = 0f64;
    let mut rng = rand::thread_rng();
    let mut weights = init_weights.clone();
    for i in 0..epochs {
        if i % 100 == 0 && !silent {
            let mut h = data * &weights;

            if fit_intercept {
                h.add_scalar_mut(intercept);
            }
            match model {
                Model::None => (),
                Model::Logit => h.apply(|x| {
                    let res = sigmoid(*x);
                    *x = res;
                }),
            };

            let error: f64 = match loss {
                LossFunction::BinCrossEntropy => binary_cross_entropy(&target, &h),
                LossFunction::MSE => mean_squared_error(&target, &h),
            };
            println!("Epoch <{}: Current Errors {}", i, error);
        }
        let batch_size = data.nrows() / n_batches;
        for _ in 0..(n_batches) {
            let j = rng.gen_range(0..n_batches);
            let remaining = data.nrows() - (j * batch_size);
            let current_batch_size = if remaining < batch_size {
                remaining
            } else {
                batch_size
            };
            let inverse_batch_size = learning_rate
                * match model {
                    Model::Logit => 1.0,
                    Model::None => 2.0,
                }
                / current_batch_size as f64;
            let batch_data = data.rows(j * batch_size, current_batch_size);
            let mut h = &batch_data * &weights;

            if fit_intercept {
                h.add_scalar_mut(intercept);
            }
            match model {
                Model::None => (),
                Model::Logit => h.apply(|x| {
                    let res = sigmoid(*x);
                    *x = res;
                }),
            };
            let errors = &h - &target.rows(j * batch_size, current_batch_size);
            let weight_updates = &batch_data.transpose() * &errors * inverse_batch_size;

            // Update weights
            weights -= weight_updates;

            // Update intercept if used
            if fit_intercept {
                intercept -= errors.sum() * inverse_batch_size;
            }
        }
    }
    (weights, intercept)
}