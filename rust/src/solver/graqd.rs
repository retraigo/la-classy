extern crate nalgebra as na;
use na::DMatrix;
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
    batch_size: usize,
    epochs: usize,
    silent: bool,
) -> f64 {
    println!("EPOCHS {}", epochs);
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y: &[f64] = std::slice::from_raw_parts(y_ptr, y_len);

    let mut rng = rand::thread_rng();

    let mut weights: DMatrix<f64> = DMatrix::zeros(1, n_features);
    for i in 0..n_features {
        weights[i] = 1.0;
    }
    let mut intercept = 0f64;

    let data = DMatrix::from_row_slice(x_len, n_features, x);
    let target = DMatrix::from_row_slice(y_len, 1, y);

    for i in 0..epochs {
        let n_batches = data.nrows() / batch_size;
        for _ in 0..(n_batches) {
            let j = rng.gen_range(0..n_batches);
            let remaining = data.nrows() - (j * batch_size);
            let current_batch_size = if remaining < batch_size {
                remaining
            } else {
                batch_size
            };
            let inverse_batch_size = 2.0 / current_batch_size as f64;
            let batch_data = data.rows(j * batch_size, current_batch_size);
            let h = batch_data
                .row_iter()
                .map(|x| {
                    let mut res = weights.dot(&x);
                    if fit_intercept {
                        res = res + intercept;
                    };
                    match convertor {
                        Convertor::None => res,
                        Convertor::Logit => sigmoid(res),
                    }
                })
                .collect::<Vec<f64>>();
            let y1 = DMatrix::from_vec(current_batch_size, 1, h);
            let errors = y1 - &target.rows(j * batch_size, current_batch_size);
            println!("Erro {:?}", errors.as_slice());
            //    println!("Errors {:?}", errors.as_slice());
            let weight_updates: Vec<f64> = (0..n_features)
                .map(|i| {
                    let x_i = batch_data.column(i);
                    learning_rate * x_i.dot(&errors) * inverse_batch_size
                })
                .collect();
            println!("Wei {:?}", weight_updates.as_slice());
            // Update weights
            weights = weights - DMatrix::from_vec(1, n_features, weight_updates);

            if fit_intercept {
                intercept -= errors.sum() * learning_rate * inverse_batch_size;
            }
        }
    }
    let res_weights = std::slice::from_raw_parts_mut(w_ptr, n_features);
    for i in 0..weights.ncols() {
        res_weights[i] = weights.column(i)[0];
    }
    println!("W {:?}", res_weights);
    intercept
}
