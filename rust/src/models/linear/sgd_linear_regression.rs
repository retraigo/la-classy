// This version uses Stochastic Gradient Descent
// instead of Ordinary Least Squares

extern crate nalgebra as na;
use na::DMatrix;
use rand::Rng;

use crate::common::functions::mean_squared_error;

pub struct SgdLinearRegressionResult {
    pub weights: DMatrix<f64>,
    pub intercept: f64,
    pub n_features: usize,
}

#[no_mangle]
pub extern "C" fn sgd_linear_regression_free_res(res: *const SgdLinearRegressionResult) {
    let _res = res.to_owned();
}
#[no_mangle]
pub unsafe extern "C" fn sgd_linear_regression(
    x_ptr: *const f64,
    y_ptr: *const f64,
    x_len: usize,
    y_len: usize,
    n_features: usize,
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
    silent: bool,
) -> isize {
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y = std::slice::from_raw_parts(y_ptr, y_len);
    let inverse_n = 1.0 / y.len() as f64;

    let mut rng = rand::thread_rng();

    let mut intercept = 0.0;
    let mut weights: DMatrix<f64> = DMatrix::zeros(1, n_features);
    for i in 0..n_features {
        weights[i] = 1.0;
    }
    let data = DMatrix::from_row_slice(x_len, n_features, x);
    let target = DMatrix::from_row_slice(y_len, 1, y);
    for i in 0..epochs {
        if i % 100 == 0 && !silent {
            let y1: Vec<f64> = data
                .row_iter()
                .map(|x| weights.dot(&x) + intercept)
                .collect();
            // Calculate MSE
            let error: f64 = mean_squared_error(y, y1.as_slice());
            println!("Epoch <{}: Current Errors {}", i, error);
        }
        let n_batches = data.nrows() / batch_size;
        for _ in 0..(n_batches) {
            let j = rng.gen_range(0..n_batches);
            let remaining = data.nrows() - (j * batch_size);
            let current_batch_size = if remaining < batch_size {
                remaining
            } else {
                batch_size
            };
            let batch_data = data.rows(j * batch_size, current_batch_size);
            let y1 = DMatrix::from_vec(
                batch_data.nrows(),
                1,
                batch_data
                    .row_iter()
                    .map(|x| weights.dot(&x) + intercept)
                    .collect(),
            );
            let errors = y1 - &target.rows(j * batch_size, current_batch_size);

            //            println!("Errors are {:?}", errors.as_slice());
            let weight_updates: Vec<f64> = (0..n_features)
                .map(|i| {
                    let x_i = batch_data.column(i);
                    learning_rate * x_i.dot(&errors) * inverse_n * 2.0
                })
                .collect();
            let intercept_update = errors.sum();
            // Update weights
            weights -= DMatrix::from_vec(1, n_features, weight_updates);

            intercept -= learning_rate * (inverse_n * 2.0) * intercept_update;
        }
    }

    let res = SgdLinearRegressionResult {
        weights,
        intercept,
        n_features,
    };
    return std::mem::transmute::<Box<SgdLinearRegressionResult>, isize>(std::boxed::Box::new(res))
        as isize;
}

#[no_mangle]
pub unsafe extern "C" fn sgd_linear_regression_predict_y(
    res: *const SgdLinearRegressionResult,
    x_ptr: *const f64,
) -> f64 {
    let _res = &*res;

    let x = std::slice::from_raw_parts(x_ptr, _res.n_features);

    let y = _res
        .weights
        .dot(&DMatrix::from_row_slice(1, _res.n_features, x));
    return y + _res.intercept;
}
