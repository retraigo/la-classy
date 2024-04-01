mod gd;
mod sag;

pub use gd::GradientDescentSolver;
use ndarray::Array2;
pub use sag::SAGSolver;

use super::regularization::Regularization;

pub enum Solver {
    GD(GradientDescentSolver),
    SAG(SAGSolver)
}

impl Solver {
    pub fn solve(
        &mut self,
        data: &Array2<f32>,
        targets: &Array2<f32>,
        epochs: usize,
        learning_rate: f32,
        n_batches: usize,
        silent: bool,
        tolerance: f32,
        patience: isize,
        regularizer: &Regularization,
    ) -> Array2<f32> {
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
        }
    }
    pub fn predict(&self, data: &Array2<f32>, weights: &Array2<f32>) -> Array2<f32> {
        match self {
            Self::GD(solver) => solver.predict(data, weights),
            Self::SAG(solver) => solver.predict(data, weights),
        }
    }
}
