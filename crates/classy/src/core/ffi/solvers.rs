use ndarray::{Array2, Array1, ArrayD, IxDyn, stack};
use ndarray_linalg::hstack;

use crate::core::{
    activation::Activation,
    loss::LossFunction,
    optimizers::Optimizer,
    regularization::Regularization,
    scheduler::Scheduler,
    solver::{GradientDescentSolver, MinibatchSGDSolver, OrdinaryLeastSquares, SGDSolver, Solver},
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
pub unsafe extern "C" fn minibatch_solver(
    scheduler: *const Scheduler,
    optimizer: *mut Optimizer,
    activation: *const Activation,
    loss: *const LossFunction,
) -> isize {
    let solver = Solver::Minibatch(MinibatchSGDSolver {
        scheduler: std::ptr::read(scheduler),
        optimizer: std::ptr::read(optimizer),
        activation: std::ptr::read(activation),
        loss: std::ptr::read(loss),
    });
    std::mem::transmute::<Box<Solver>, isize>(std::boxed::Box::new(solver))
}

#[no_mangle]
pub unsafe extern "C" fn sgd_solver(
    scheduler: *const Scheduler,
    optimizer: *mut Optimizer,
    activation: *const Activation,
    loss: *const LossFunction,
) -> isize {
    let solver = Solver::SGD(SGDSolver {
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
    epochs: usize,
    learning_rate: f64,
    fit_intercept: bool,
    n_batches: usize,
    silent: bool,
    regularizer: *const Regularization,
    solver: *mut Solver,
) {
    let x = unsafe { std::slice::from_raw_parts(data_ptr, n_samples * n_features) };
    let y = unsafe { std::slice::from_raw_parts(target_ptr, n_samples) };

    let weight_buffer = unsafe {
        std::slice::from_raw_parts_mut(
            weights_ptr,
            n_features + if fit_intercept { 1 } else { 0 },
        )
    };

    let mut data = Array2::from_shape_vec([n_samples, n_features], Vec::from(x)).unwrap();
    if fit_intercept {
        let new_col = Array1::from_elem(n_samples, 1f64);
        let _ = data.push_column(new_col.view());
    }

    let targets = Array1::from_shape_vec(n_samples, Vec::from(y)).unwrap();

    let weights = (*solver).solve(
        &data,
        &targets,
        epochs,
        learning_rate,
        n_batches,
        silent,
        &*regularizer,
    );
    for i in 0..weights.len() {
        weight_buffer[i] = weights[i];
    }
}
