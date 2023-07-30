extern crate nalgebra as na;
use crate::common::types::{LossFunction, Model, Optimizer, OptimizerFFI};
use na::{DMatrix, DVector};

use crate::optimizer::optimize::optimize;

#[no_mangle]
pub unsafe extern "C" fn gradient_descent(
    w_ptr: *mut f64,
    x_ptr: *const f64,
    y_ptr: *const f64,
    x_len: usize,
    y_len: usize,
    n_features: usize,
    loss: LossFunction,
    model: Model,
    optimizer: OptimizerFFI,
    adam_options: *const f64,
    fit_intercept: bool,
    learning_rate: f64,
    n_batches: usize,
    epochs: usize,
    silent: bool,
) -> f64 {
    println!("EPOCHS {}", epochs);
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y: &[f64] = std::slice::from_raw_parts(y_ptr, y_len);

    let weights: DVector<f64> = DVector::from_element(n_features, 1.0);

    let data = DMatrix::from_row_slice(x_len, n_features, x);
    let target = DVector::from_column_slice(y);

    let (weights, intercept) = optimize(
        &data,
        &target,
        &weights,
        loss,
        model,
        match optimizer {
            OptimizerFFI::Adam => {
                let adam_opt = std::slice::from_raw_parts(adam_options, 3);
                Optimizer::Adam {
                    learning_rate,
                    beta1: adam_opt[0],
                    beta2: adam_opt[1],
                    epsilon: adam_opt[2],
                    t: 1,
                }
            }
            OptimizerFFI::MinibatchSGD => Optimizer::MinibatchSGD {
                learning_rate,
                n_batches,
            },
            OptimizerFFI::SGD => Optimizer::SGD { learning_rate },
            OptimizerFFI::GD => Optimizer::GD { learning_rate },
        },
        fit_intercept,
        epochs,
        silent,
    );

    let res_weights = std::slice::from_raw_parts_mut(w_ptr, n_features);
    for i in 0..weights.nrows() {
        res_weights[i] = weights.row(i)[0];
    }
    intercept
}
