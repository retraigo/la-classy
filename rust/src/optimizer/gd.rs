/**
 * Plain gradient descent
 */
extern crate nalgebra as na;
use crate::common::{
    functions::{binary_cross_entropy, mean_squared_error, sigmoid},
    scheduler::{get_learning_rate, LearningRateScheduler},
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
    c: f64,
    epochs: usize,
    silent: bool,
    learning_rate: f64,
    scheduler: LearningRateScheduler,
) -> (DVector<f64>, f64) {
    let mut intercept = 0f64;
    if fit_intercept {
        intercept = target.mean();
    }
    let mut weights = init_weights.clone();
    let mut eta = learning_rate;
    let inverse_n = match model {
        Model::Logit => 1.0,
        Model::None => 2.0,
    } / data.nrows() as f64;
    
    for i in 0..epochs {
        eta = get_learning_rate(&scheduler, eta, i);

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
        let weights_l1 = c * &weights.map(|w| if w >= 0.0 { 1.0 } else { -1.0 });

        // Update weights
        weights -= (weight_updates + weights_l1) * eta;

        // Update intercept if used
        if fit_intercept {
            let intercept_l1 = c * if intercept >= 0.0 { 1.0 } else { -1.0 };
            intercept -= ((errors.sum() * inverse_n) + intercept_l1) * eta;
        }
    }
    (weights, intercept)
}
