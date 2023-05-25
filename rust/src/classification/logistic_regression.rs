extern crate nalgebra as na;

use na::{DMatrix, DVector};

pub struct LogisticRegressionResult {
    pub weights: DMatrix<f32>,
    pub n_features: usize,
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[no_mangle]
pub unsafe extern "C" fn logistic_regression(
    x_ptr: *const f32,
    y_ptr: *const f32,
    x_len: usize,
    y_len: usize,
    n_features: usize,
    epochs: usize,
    silent: bool,
) -> isize {
    println!("EPOCHS {}", epochs);
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y = std::slice::from_raw_parts(y_ptr, y_len);

    let mut weights: DMatrix<f32> = DMatrix::zeros(1, n_features);
    for i in 0..n_features {
        weights[i] = 1.0;
    }

    let mut data = DMatrix::from_row_slice(x_len, n_features, x);

    for i in 0..epochs {
        let hypotheses: Vec<f32> = data
            .row_iter_mut()
            .map(|x| sigmoid(weights.dot(&x)))
            .collect();
        if i % 100 == 0 && !silent {
            let error: f32 = (0..x_len)
                .map(|i| {
                    let h_i = hypotheses.get(i).unwrap();
                    let y_i = y[i];

                    return (y_i * h_i.log2() + (1.0 - y_i) * (1.0 - h_i).log2()) as f32;
                })
                .sum::<f32>()
                / x_len as f32;
            println!("Epoch <{}: Current Errors {}", i, error);
        }

        let errors: Vec<f32> = (0..x_len)
            .map(|i| hypotheses.get(i).unwrap() - y[i])
            .collect();

        let err = DVector::from_vec(errors);

        let weight_updates: Vec<f32> = (0..n_features)
            .map(|i| {
                let x_i = data.column(i);
                0.001 * x_i.dot(&err)
            })
            .collect();

        // Update weights
        weights = weights - DMatrix::from_vec(1, n_features, weight_updates);
        if !silent {
            println!("Finished epoch {}", i);
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
    x_ptr: *const f32,
) -> f32 {
    let _res = &*res;

    let x = std::slice::from_raw_parts(x_ptr, _res.n_features);

    let y = _res
        .weights
        .dot(&DMatrix::from_row_slice(1, _res.n_features, x));
    return y;
}
#[no_mangle]
pub extern "C" fn logistic_regression_free_res(res: *const LogisticRegressionResult) {
    let _res = res.to_owned();
}
