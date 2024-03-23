use ndarray::{Array2, Axis};

use crate::core::{
    activation::Activation,
    loss::LossFunction,
    optimizers::Optimizer,
    regularization::Regularization,
    scheduler::Scheduler,
    solver::{GradientDescentSolver, SAGSolver, Solver},
};

#[no_mangle]
pub unsafe extern "C" fn gd_solver(
    scheduler: *const Scheduler,
    optimizer: *mut Optimizer,
    activation: *const Activation,
    loss: *const LossFunction,
) -> isize {
    let solver = Solver::GD(GradientDescentSolver {
        scheduler: std::ptr::read(scheduler),
        optimizer: std::ptr::read(optimizer),
        activation: std::ptr::read(activation),
        loss: std::ptr::read(loss),
    });
    std::mem::transmute::<Box<Solver>, isize>(std::boxed::Box::new(solver))
}

#[no_mangle]
pub unsafe extern "C" fn sag_solver(
    scheduler: *const Scheduler,
    optimizer: *mut Optimizer,
    activation: *const Activation,
    loss: *const LossFunction,
) -> isize {
    let solver = Solver::SAG(SAGSolver {
        scheduler: std::ptr::read(scheduler),
        optimizer: std::ptr::read(optimizer),
        activation: std::ptr::read(activation),
        loss: std::ptr::read(loss),
    });
    std::mem::transmute::<Box<Solver>, isize>(std::boxed::Box::new(solver))
}

#[no_mangle]
pub unsafe extern "C" fn ols_solver() -> isize {
    1
}

#[no_mangle]
pub unsafe extern "C" fn solve(
    weights_ptr: *mut f64,
    data_ptr: *const f64,
    target_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    n_categories: usize,
    epochs: usize,
    learning_rate: f64,
    fit_intercept: bool,
    n_batches: usize,
    silent: bool,
    tolerance: f64,
    patience: isize,
    regularizer: *const Regularization,
    solver: *mut Solver,
) {
    let x = unsafe { std::slice::from_raw_parts(data_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(target_ptr, n_samples * n_categories) };

    let weight_buffer = unsafe {
        std::slice::from_raw_parts_mut(
            weights_ptr,
            (n_features + if fit_intercept { 1 } else { 0 }) * n_categories,
        )
    };

    let mut data: Array2<f64> =
        Array2::from_shape_vec((n_samples, n_features), x.to_vec()).unwrap();
    println!("BEFORE {:?}", data.shape());
    if fit_intercept {
        let new_col: Array2<f64> = Array2::ones((data.nrows(), 1));
        data = ndarray::concatenate(Axis(0), &[new_col.view(), data.view()]).unwrap();
    }
    println!("BEFORE {:?}", data.shape());

    let targets = Array2::from_shape_vec((n_samples, n_categories), y.to_vec()).unwrap();
    let weights = (*solver).solve(
        &data,
        &targets,
        epochs,
        learning_rate,
        n_batches,
        silent,
        tolerance,
        patience,
        &*regularizer,
    );
    for (row, i) in weights.axis_iter(Axis(0)).zip(0..weights.nrows()) {
        for (item, j) in row.iter().zip(0..weights.ncols()) {
            weight_buffer[i * weights.ncols() + j] = *item;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn predict(
    res_ptr: *mut f64,
    weights_ptr: *const f64,
    data_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    n_weights: usize,
    fit_intercept: bool,
    solver: *mut Solver,
) {
    let x = unsafe { std::slice::from_raw_parts(data_ptr, n_samples * n_features) };
    let weights: &[f64] = unsafe {
        std::slice::from_raw_parts(
            weights_ptr,
            (n_features + if fit_intercept { 1 } else { 0 }) * n_weights,
        )
    };
    let mut data = Array2::from_shape_vec((n_samples, n_features), x.to_vec()).unwrap();
    let w = Array2::from_shape_vec(
        (n_features + if fit_intercept { 1 } else { 0 }, n_weights),
        weights.to_vec(),
    )
    .unwrap();

    if fit_intercept {
        let new_col: Array2<f64> = Array2::ones((data.nrows(), 1));
        data = ndarray::concatenate(Axis(1), &[new_col.view(), data.view()]).unwrap();
    }
    let res = (*solver).predict(&data, &w);
    let res_buffer = unsafe { std::slice::from_raw_parts_mut(res_ptr, n_samples * n_weights) };
    for (row, i) in res.axis_iter(Axis(0)).zip(0..res.nrows()) {
        for (item, j) in row.iter().zip(0..res.ncols()) {
            res_buffer[i * res.ncols() + j] = *item;
        }
    }
}
