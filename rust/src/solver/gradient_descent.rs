extern crate nalgebra as na;
use crate::common::{
    scheduler::{LearningRateScheduler, LearningRateSchedulerFFI},
    types::{LossFunction, Model, Optimizer, OptimizerFFI},
};
use na::{DMatrix, DVector};

use crate::optimizer::optimize::optimize;

#[no_mangle]
pub unsafe extern "C" fn gradient_descent(
    w_ptr: *mut f64,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    loss: LossFunction,
    model: Model,
    c: f64,
    optimizer: OptimizerFFI,
    optimizer_options: *const f64,
    scheduler: LearningRateSchedulerFFI,
    scheduler_options: *const f64,
    fit_intercept: bool,
    learning_rate: f64,
    epochs: usize,
    silent: bool,
) -> f64 {
    println!("EPOCHS {}", epochs);
    let x = std::slice::from_raw_parts(x_ptr, n_samples * n_features);
    let y: &[f64] = std::slice::from_raw_parts(y_ptr, n_samples);

    let weights: DVector<f64> = DVector::from_element(n_features, 1.0);

    let data = DMatrix::from_row_slice(n_samples, n_features, x);
    let target = DVector::from_column_slice(y);

    let (weights, intercept) = optimize(
        &data,
        &target,
        &weights,
        loss,
        model,
        c,
        match optimizer {
            OptimizerFFI::Adam => {
                let optimizer_opt = std::slice::from_raw_parts(optimizer_options, 3);
                Optimizer::Adam {
                    learning_rate,
                    beta1: optimizer_opt[0],
                    beta2: optimizer_opt[1],
                    epsilon: optimizer_opt[2],
                    t: 1,
                }
            }
            OptimizerFFI::MinibatchSGD => {
                let optimizer_opt = std::slice::from_raw_parts(optimizer_options, 1);
                Optimizer::MinibatchSGD {
                    learning_rate,
                    n_batches: optimizer_opt[0] as usize,
                }
            }
            OptimizerFFI::SGD => Optimizer::SGD { learning_rate },
            OptimizerFFI::GD => Optimizer::GD { learning_rate },
        },
        match scheduler {
            LearningRateSchedulerFFI::None => LearningRateScheduler::None,
            LearningRateSchedulerFFI::DecayScheduler => {
                let scheduler_opt = std::slice::from_raw_parts(scheduler_options, 1);
                LearningRateScheduler::DecayScheduler {
                    rate: scheduler_opt[0],
                }
            }
            LearningRateSchedulerFFI::AnnealingScheduler => {
                let scheduler_opt = std::slice::from_raw_parts(scheduler_options, 2);
                LearningRateScheduler::AnnealingScheduler {
                    rate: scheduler_opt[0],
                    step_size: scheduler_opt[1] as usize,
                }
            }
            LearningRateSchedulerFFI::OneCycleScheduler => {
                let scheduler_opt = std::slice::from_raw_parts(scheduler_options, 3);
                LearningRateScheduler::OneCycleScheduler {
                    initial_lr: scheduler_opt[0],
                    max_lr: scheduler_opt[1],
                    cycle_steps: scheduler_opt[2] as usize,
                }
            }
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
