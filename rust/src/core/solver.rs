use nalgebra::{DMatrix, DVector};

use crate::types::{ModelConfig, Solver};

use super::{
    gradient_descent::{
        gd::gradient_descent, minibatch_sgd::minibatch_stochastic_gradient_descent,
        sgd::stochastic_gradient_descent,
    },
    least_squares::ols::ordinary_least_squares,
};

#[no_mangle]
pub extern "C" fn solve(
    weights_ptr: *mut f64,
    data_ptr: *const f64,
    target_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    config_ptr: *const u8,
    config_len: usize,
    solver: Solver,
) {
    let buffer = unsafe { std::slice::from_raw_parts(config_ptr, config_len) };
    let json = std::str::from_utf8(&buffer[0..config_len]).unwrap();
    let config: ModelConfig = match serde_json::from_str(&json) {
        Ok(data) => data,
        Err(err) => {
            println!("{:?} {}", err, json);
            panic!("HHHH");
        }
    };

    let x = unsafe { std::slice::from_raw_parts(data_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(target_ptr, n_samples) };

    let weight_buffer = unsafe {
        std::slice::from_raw_parts_mut(
            weights_ptr,
            n_features + if config.fit_intercept { 1 } else { 0 },
        )
    };

    let mut data = DMatrix::from_row_slice(n_samples, n_features, x);
    if config.fit_intercept {
        data = data.insert_column(n_features, 1.0);
    }
    println!("SHAP {:?}", data.shape());
    let targets = DVector::from_column_slice(y);

    let weights = match solver {
        Solver::OLS => ordinary_least_squares(&config, &data, &targets),
        Solver::SGD => stochastic_gradient_descent(&config, &data, &targets),
        Solver::GD => gradient_descent(&config, &data, &targets),
        Solver::Minibatch => minibatch_stochastic_gradient_descent(&config, &data, &targets),
    };
    println!("Wei {:?}", weights.as_slice());
    for i in 0..weights.len() {
        weight_buffer[i] = weights[i];
    }
}
