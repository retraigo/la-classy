/**
 * Plain gradient descent
 */

extern crate nalgebra as na;
use crate::common::{
    functions::{binary_cross_entropy, mean_squared_error, sigmoid},
    types::{LossFunction, Model},
};
use na::{DMatrix, DVector};
pub fn gradient_descent_optimizer(
    data: &DMatrix<f64>,
    target: &DVector<f64>,
    init_weights: &DVector<f64>,
    loss: LossFunction,
    model: Model,
    fit_intercept: bool,
    epochs: usize,
    silent: bool,
    learning_rate: f64,
) -> (DVector<f64>, f64) {
    let mut intercept = 0f64;
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

        let inverse_n = match model {
            Model::Logit => 1.0,
            Model::None => 2.0,
        } / data.nrows() as f64;
        let mut h = data * &weights;

        match model {
            Model::None => (),
            Model::Logit => h.apply(|x| {
                let res = sigmoid(*x);
                *x = res;
            }),
        };
        let errors = h - target;
        let weight_updates = &data.transpose() * &errors * inverse_n;
        // Update weights
        weights -= weight_updates * learning_rate;

        // Update intercept if used
        if fit_intercept {
            intercept -= errors.sum() * inverse_n * learning_rate;
        }
    }
    (weights, intercept)
}
