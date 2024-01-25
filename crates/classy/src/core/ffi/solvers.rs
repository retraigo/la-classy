use nalgebra::DMatrix;

use crate::core::{
    activation::Activation,
    loss::LossFunction,
    optimizers::Optimizer,
    regularization::Regularization,
    scheduler::Scheduler,
    solver::{GradientDescentSolver, SAGSolver, OrdinaryLeastSquares, Solver},
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
    let solver = Solver::OLS(OrdinaryLeastSquares);
    std::mem::transmute::<Box<Solver>, isize>(std::boxed::Box::new(solver))
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
    regularizer: *const Regularization,
    solver: *mut Solver,
) {
    let x = unsafe { std::slice::from_raw_parts(data_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(target_ptr, n_samples * n_categories) };

    let weight_buffer = unsafe {
        std::slice::from_raw_parts_mut(weights_ptr, (n_features + if fit_intercept { 1 } else { 0 }) * n_categories)
    };

    let mut data = DMatrix::from_row_slice(n_samples, n_features, x);
    if fit_intercept {
        data = data.insert_column(0, 1.0);
    }

    let targets = DMatrix::from_row_slice(n_samples, n_categories, y);

    let weights = (*solver).solve(
        &data,
        &targets,
        epochs,
        learning_rate,
        n_batches,
        silent,
        &*regularizer,
    );
    for (row, i) in weights.row_iter().zip(0..weights.nrows()) {
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
    let weights: &[f64] =
        unsafe { std::slice::from_raw_parts(weights_ptr, n_features * n_weights) };
    let mut data = DMatrix::from_row_slice(n_samples, n_features, x);
    let w = DMatrix::from_row_slice(n_features, n_weights, weights);

    if fit_intercept {
        data = data.insert_column(0, 1.0);
    }
    let res = (*solver).predict(&data, &w);
    let res_buffer = unsafe { std::slice::from_raw_parts_mut(res_ptr, n_samples * n_weights) };
    for (row, i) in res.row_iter().zip(0..res.nrows()) {
        for (item, j) in row.iter().zip(0..res.ncols()) {
            res_buffer[i * res.ncols() + j] = *item;
        }
    }
}
