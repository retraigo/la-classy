extern crate nalgebra as na;
use na::DMatrix;
use rand::Rng;

use crate::common::functions::{logit, sigmoid};

pub struct LogisticRegressionResult {
    pub weights: DMatrix<f64>,
    pub n_features: usize,
}

#[no_mangle]
pub unsafe extern "C" fn logistic_regression(
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
    println!("EPOCHS {}", epochs);
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y: &[f64] = std::slice::from_raw_parts(y_ptr, y_len);

    let mut rng = rand::thread_rng();

    let mut weights: DMatrix<f64> = DMatrix::zeros(1, n_features);
    for i in 0..n_features {
        weights[i] = 1.0;
    }

    let data = DMatrix::from_row_slice(x_len, n_features, x);
    let target = DMatrix::from_row_slice(y_len, 1, y);

    for i in 0..epochs {
        if i % 100 == 0 && !silent {
            let y1: Vec<f64> = data.row_iter().map(|x| sigmoid(weights.dot(&x))).collect();
            let error: f64 = logit(&y, &y1);
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
            let h = batch_data
                .row_iter()
                .map(|x| sigmoid(weights.dot(&x)))
                .collect::<Vec<f64>>();
            let y1 = DMatrix::from_vec(current_batch_size, 1, h);
            let errors = y1 - &target.rows(j * batch_size, current_batch_size);
            let weight_updates: Vec<f64> = (0..n_features)
                .map(|i| {
                    let x_i = batch_data.column(i);
                    learning_rate * x_i.dot(&errors)
                })
                .collect();

            // Update weights
            weights = weights - DMatrix::from_vec(1, n_features, weight_updates);
        }
    }
    let res = LogisticRegressionResult {
        weights,
        n_features,
    };
    return std::mem::transmute::<Box<LogisticRegressionResult>, isize>(std::boxed::Box::new(res))
        as isize;
}

#[no_mangle]
pub unsafe extern "C" fn logistic_regression_predict_y(
    res: *const LogisticRegressionResult,
    x_ptr: *const f64,
) -> f64 {
    let _res = &*res;

    let x = std::slice::from_raw_parts(x_ptr, _res.n_features);

    let y = _res
        .weights
        .dot(&DMatrix::from_row_slice(1, _res.n_features, x));
    return sigmoid(y);
}

#[no_mangle]
pub unsafe extern "C" fn logistic_regression_confusion_matrix(
    res: *const LogisticRegressionResult,
    x_ptr: *const f64,
    y_ptr: *const f64,
    x_len: usize,
    y_len: usize,
    matrix_ptr: *mut f64,
) {
    let _res = &*res;

    let x = std::slice::from_raw_parts(x_ptr, _res.n_features * x_len);
    let y = std::slice::from_raw_parts(y_ptr, y_len);
    let data = DMatrix::from_row_slice(x_len, _res.n_features, x);
    let res = std::slice::from_raw_parts_mut(matrix_ptr, 4);

    for i in 0..x_len {
        let row = data.row(i);
        let target = y[i];
        let guess = if sigmoid(_res.weights.dot(&row)) < 0.5 {
            0
        } else {
            1
        };
        // True Positive, False Negative, False Positive, True Negative
        match (guess, target as u8) {
            (0, 0) => res[3] += 1.0,
            (0, 1) => res[1] += 1.0,
            (1, 0) => res[2] += 1.0,
            (1, 1) => res[0] += 1.0,
            (_, _) => (),
        }
    }
}

#[no_mangle]
pub extern "C" fn logistic_regression_free_res(res: *const LogisticRegressionResult) {
    let _res = res.to_owned();
}
