extern crate nalgebra as na;
use crate::common::{
    functions::{binary_cross_entropy, mean_squared_error, sigmoid},
    types::{LossFunction, Model},
};
use na::{DMatrix, DVector};

pub fn adam_optimizer(
    data: &DMatrix<f64>,
    target: &DVector<f64>,
    init_weights: &DVector<f64>,
    loss: LossFunction,
    model: Model,
    fit_intercept: bool,
    epochs: usize,
    silent: bool,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
) -> (DVector<f64>, f64) {
    let mut weight_m = DVector::zeros(data.ncols());
    let mut weight_v = DVector::zeros(data.ncols());

    let mut intercept_m = 0f64;
    let mut intercept_v = 0f64;
    let mut tv = t;

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

        let beta1_pow_t = beta1.powi(tv as i32);
        let beta2_pow_t = beta2.powi(tv as i32);

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

        // Update biased first moment estimate
        weight_m *= beta1;
        weight_m += (1.0 - beta1) * &weight_updates;

        // Update biased second raw moment estimate
        weight_v *= beta2;
        weight_v += (1.0 - beta2) * &weight_updates.component_mul(&weight_updates);

        let m_hat = &weight_m / (1.0 - beta1_pow_t);
        let v_hat = &weight_v / (1.0 - beta2_pow_t);

        // Perform element-wise division
        let updates = &m_hat.zip_map(&v_hat.map(|x| x.sqrt()).add_scalar(epsilon), |x, y| x / y);
        weights = weights - updates * learning_rate;

        if fit_intercept {
            let intercept_updates = errors.sum() * inverse_n;
            intercept_m *= beta1;
            intercept_m += (1.0 - beta1) * intercept_updates;

            intercept_v *= beta2;
            intercept_v += (1.0 - beta2) * (intercept_updates * intercept_updates);

            let intercept_m_hat = intercept_m / (1.0 - beta1_pow_t);
            let intercept_v_hat = intercept_v / (1.0 - beta2_pow_t);

            intercept -= (learning_rate / (intercept_v_hat.sqrt() + epsilon)) * intercept_m_hat;
        }

        tv += 1;
    }
    (weights, intercept)
}
