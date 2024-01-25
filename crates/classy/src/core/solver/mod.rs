mod gd;
mod ols;
mod sag;

pub use gd::GradientDescentSolver;
use nalgebra::DMatrix;
pub use ols::OrdinaryLeastSquares;
pub use sag::SAGSolver;

use super::regularization::Regularization;

pub enum Solver {
    GD(GradientDescentSolver),
    OLS(OrdinaryLeastSquares),
    SAG(SAGSolver)
}

impl Solver {
    pub fn solve(
        &mut self,
        data: &DMatrix<f64>,
        targets: &DMatrix<f64>,
        epochs: usize,
        learning_rate: f64,
        n_batches: usize,
        silent: bool,
        tolerance: f64,
        patience: isize,
        regularizer: &Regularization,
    ) -> DMatrix<f64> {
        match self {
            Self::GD(solver) => solver.train(
                data,
                targets,
                epochs,
                learning_rate,
                n_batches,
                silent,
                tolerance,
                patience,
                regularizer,
            ),
            Self::SAG(solver) => solver.train(
                data,
                targets,
                epochs,
                learning_rate,
                n_batches,
                silent,
                tolerance,
                patience,
                regularizer,
            ),
            Self::OLS(solver) => solver.train(data, targets, silent),
        }
    }
    pub fn predict(&self, data: &DMatrix<f64>, weights: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Self::GD(solver) => solver.predict(data, weights),
            Self::OLS(solver) => solver.predict(data, weights),
            Self::SAG(solver) => solver.predict(data, weights),
        }
    }
}
