extern crate nalgebra as na;
use crate::common::functions::sigmoid;
use na::{DMatrix, DVector};
// use std::collections::HashSet;
pub struct LogisticRegressionResult {
    pub weights: DMatrix<f64>,
    pub n_features: usize,
}

#[no_mangle]
pub unsafe extern "C" fn logistic_regression(
    x_ptr: *const f64,
    y_ptr: *const u8,
    x_len: usize,
    y_len: usize,
    n_features: usize,
    learning_rate: f64,
    epochs: usize,
    silent: bool,
) -> isize {
    println!("EPOCHS {}", epochs);
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y = std::slice::from_raw_parts(y_ptr, y_len);
    // let classes: HashSet<&u32> = HashSet::from_iter(y.into_iter());
    // let n_classes = classes.len();

    let mut weights: DMatrix<f64> = DMatrix::zeros(1, n_features);
    for i in 0..n_features {
        weights[i] = 1.0;
    }

    let mut data = DMatrix::from_row_slice(x_len, n_features, x);

    for i in 0..epochs {
        let hypotheses: Vec<f64> = data
            .row_iter_mut()
            .map(|x| sigmoid(weights.dot(&x)))
            .collect();
        if i % 100 == 0 && !silent {
            // Calculate log-like cost for all parameters w
            // logL(w) = ∑ (yi * log(hi) + (1 − yi) * log(1 − hi))
            let error: f64 = (0..x_len)
                .map(|i| {
                    let h_i = hypotheses.get(i).unwrap();
                    let y_i = y[i] as f64;
                    y_i * h_i.log2() + (1.0 - y_i) * (1.0 - h_i).log2()
                })
                .sum::<f64>()
                / x_len as f64;
            println!("Epoch <{}: Current Errors {}", i, error);
        }

        let errors: Vec<f64> = (0..x_len)
            .map(|i| hypotheses.get(i).unwrap() - (y[i] as f64))
            .collect();

        let err = DVector::from_vec(errors);

        let weight_updates: Vec<f64> = (0..n_features)
            .map(|i| {
                let x_i = data.column(i);
                learning_rate * x_i.dot(&err)
            })
            .collect();
        println!("Weight updates are {:?}", weight_updates.as_slice());


        // Update weights
        weights = weights - DMatrix::from_vec(1, n_features, weight_updates);
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
    y_ptr: *const u8,
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
        match (guess, target) {
            (0, 0) => res[3] += 1.0,
            (0, 1) => res[1] += 1.0,
            (1, 0) => res[2] += 1.0,
            (1, 1) => res[0] += 1.0,
            (_, _) => ()
        }
    }
}

#[no_mangle]
pub extern "C" fn logistic_regression_free_res(res: *const LogisticRegressionResult) {
    let _res = res.to_owned();
}
