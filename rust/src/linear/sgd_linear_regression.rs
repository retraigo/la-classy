// This version uses Stochastic Gradient Descent
// instead of Ordinary Least Squares

extern crate nalgebra as na;

use crate::common::functions::mean_squared_error;
use na::{DMatrix, Matrix};

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
    epochs: usize,
    silent: bool,
) -> isize {
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y = std::slice::from_raw_parts(y_ptr, y_len);
    let inverse_n = 1.0 / y.len() as f64;
    // let classes: HashSet<&u32> = HashSet::from_iter(y.into_iter());
    // let n_classes = classes.len();

    let mut intercept = 0.0;
    let mut weights: DMatrix<f64> = DMatrix::zeros(1, n_features);
    for i in 0..n_features {
        weights[i] = 1.0;
    }

    let mut data = DMatrix::from_row_slice(x_len, n_features, x);
    data = data.try_normalize(1.0e-6).unwrap();

    for i in 0..epochs {
        let y1: Vec<f64> = data
            .row_iter_mut()
            .map(|x| weights.dot(&x) + intercept)
            .collect();

        if i % 100 == 0 && !silent {
            // Calculate MSE
            let error: f64 = mean_squared_error(y, y1.as_slice());
            println!("Epoch <{}: Current Errors {}", i, error);
        }
        let errors = DMatrix::from_vec(x_len, 1, (0..x_len).map(|i| y1[i] - (y[i] as f64)).collect());
        //        println!("Errors are {:?}", errors.as_slice());
        let mut weight_updates = DMatrix::zeros(1, n_features);
        let mut intercept_update = 0.0;
        for i in 0..errors.len() {
            let x_i = data.row(i);
            let res = x_i * errors[i];
            weight_updates += res;
            intercept_update += errors[i];
        }

        // Update weights
        weights = weights - learning_rate * (inverse_n * 2.0) * weight_updates;

        intercept = intercept - learning_rate * (inverse_n * 2.0) * intercept_update;
    }
    println!("Weights: {:?}", &weights.as_slice());

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
