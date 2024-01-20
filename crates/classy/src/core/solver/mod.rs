mod gd;
mod ols;

pub use gd::GradientDescentSolver;
use nalgebra::DMatrix;
pub use ols::OrdinaryLeastSquares;

use super::regularization::Regularization;

pub enum Solver {
    GD(GradientDescentSolver),
    OLS(OrdinaryLeastSquares),
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
                regularizer,
            ),
            Self::OLS(solver) => solver.train(data, targets, silent),
        }
    }
    pub fn predict(&self, data: &DMatrix<f64>, weights: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Self::GD(solver) => solver.predict(data, weights),
            Self::OLS(solver) => solver.predict(data, weights),
        }
    }
}
