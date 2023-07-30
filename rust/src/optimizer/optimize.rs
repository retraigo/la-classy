extern crate nalgebra as na;

use na::{DMatrix, DVector};

use crate::{
    common::{types::{LossFunction, Model, Optimizer}, scheduler::LearningRateScheduler},
    optimizer::{
        adam::adam_optimizer, gd::gradient_descent_optimizer,
        minibatch_sgd::minibatch_stochastic_gradient_descent_optimizer, sgd::stochastic_gradient_descent_optimizer,
    },
};

pub fn optimize(
    data: &DMatrix<f64>,
    target: &DVector<f64>,
    init_weights: &DVector<f64>,
    loss: LossFunction,
    model: Model,
    c: f64,
    optimizer: Optimizer,
    scheduler: LearningRateScheduler,
    fit_intercept: bool,
    epochs: usize,
    silent: bool,
) -> (DVector<f64>, f64) {
    match optimizer {
        Optimizer::Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t,
        } => adam_optimizer(
            data,
            target,
            init_weights,
            loss,
            model,
            fit_intercept,
            c,
            epochs,
            silent,
            learning_rate,
            scheduler,
            beta1,
            beta2,
            epsilon,
            t,
        ),
        Optimizer::SGD { learning_rate } => stochastic_gradient_descent_optimizer(
            data,
            target,
            init_weights,
            loss,
            model,
            fit_intercept,
            c,
            epochs,
            silent,
            learning_rate,
            scheduler
        ),
        Optimizer::MinibatchSGD {
            learning_rate,
            n_batches,
        } => minibatch_stochastic_gradient_descent_optimizer(
            data,
            target,
            init_weights,
            loss,
            model,
            fit_intercept,
            c,
            epochs,
            silent,
            learning_rate,
            scheduler,
            n_batches,
        ),
        Optimizer::GD { learning_rate } => gradient_descent_optimizer(
            data,
            target,
            init_weights,
            loss,
            model,
            fit_intercept,
            c,
            epochs,
            silent,
            learning_rate,
            scheduler
        ),
    }
}
