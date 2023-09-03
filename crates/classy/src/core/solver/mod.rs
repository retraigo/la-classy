mod gd;
mod minibatch;
mod ols;
mod sgd;
mod sag;

pub use gd::GradientDescentSolver;
pub use minibatch::MinibatchSGDSolver;
use nalgebra::{DMatrix, DVector};
pub use ols::OrdinaryLeastSquares;
pub use sgd::SGDSolver;
pub use sag::SAGSolver;

use super::regularization::Regularization;

pub enum Solver {
    GD(GradientDescentSolver),
    Minibatch(MinibatchSGDSolver),
    SGD(SGDSolver),
    SAG(SAGSolver),
    OLS(OrdinaryLeastSquares),
}

impl Solver {
    pub fn solve(
        &mut self,
        data: &DMatrix<f64>,
        targets: &DVector<f64>,
        epochs: usize,
        learning_rate: f64,
        n_batches: usize,
        silent: bool,
        regularizer: &Regularization,
    ) -> DVector<f64> {
        match self {
            Self::GD(solver) => solver.train(
                data,
                targets,
                epochs,
                learning_rate,
                silent,
                regularizer,
            ),
            Self::Minibatch(solver) => solver.train(
                data,
                targets,
                epochs,
                learning_rate,
                n_batches,
                silent,
                regularizer,
            ),
            Self::SGD(solver) => solver.train(
                data,
                targets,
                epochs,
                learning_rate,
                silent,
                regularizer,
            ),
            Self::SAG(solver) => solver.train(
                data,
                targets,
                epochs,
                learning_rate,
                silent,
                regularizer,
            ),
            Self::OLS(solver) => solver.train(
                data,
                targets,
                silent,
            ),
        }
    }
}
