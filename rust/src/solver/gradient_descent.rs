/**
 * Stochastic Gradient Descent
 */
extern crate nalgebra as na;
use na::{DMatrix, DVector};
use rand::Rng;

use crate::common::{
    functions::{binary_cross_entropy, mean_squared_error, sigmoid},
    types::{Convertor, LossFunction},
};

#[no_mangle]
pub unsafe extern "C" fn linear_gradient_descent(
    w_ptr: *mut f64,
    x_ptr: *const f64,
    y_ptr: *const f64,
    x_len: usize,
    y_len: usize,
    n_features: usize,
    loss: LossFunction,
    convertor: Convertor,
    fit_intercept: bool,
    learning_rate: f64,
    n_batches: usize,
    epochs: usize,
    silent: bool,
) -> f64 {
    println!("EPOCHS {}", epochs);
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y: &[f64] = std::slice::from_raw_parts(y_ptr, y_len);

    let mut rng = rand::thread_rng();

    let mut weights: DVector<f64> = DVector::from_element(n_features, 1.0);

    let mut intercept = 0f64;

    let data = DMatrix::from_row_slice(x_len, n_features, x);
    let target = DVector::from_column_slice(y);

    for i in 0..epochs {
        if i % 100 == 0 && !silent {
            let mut h = &data * &weights;

            if fit_intercept {
                h.add_scalar_mut(intercept);
            }
            match convertor {
                Convertor::None => (),
                Convertor::Logit => h.apply(|x| {
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
                * match convertor {
                    Convertor::Logit => 2.0,
                    Convertor::None => 2.0,
                }
                / current_batch_size as f64;
            let batch_data = data.rows(j * batch_size, current_batch_size);
            let mut h = &batch_data * &weights;

            if fit_intercept {
                h.add_scalar_mut(intercept);
            }
            match convertor {
                Convertor::None => (),
                Convertor::Logit => h.apply(|x| {
                    let res = sigmoid(*x);
                    *x = res;
                }),
            };
            let errors = &h - &target.rows(j * batch_size, current_batch_size);
            let weight_updates = &batch_data.transpose() * &errors * inverse_batch_size;

            // Update weights
            weights = weights - weight_updates;

            // Update intercept if used
            if fit_intercept {
                intercept -= errors.sum() * inverse_batch_size;
            }
        }
    }
    let res_weights = std::slice::from_raw_parts_mut(w_ptr, n_features);
    for i in 0..weights.nrows() {
        res_weights[i] = weights.row(i)[0];
    }
    intercept
}
