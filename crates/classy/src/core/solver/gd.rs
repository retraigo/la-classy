use nalgebra::DMatrix;

use crate::core::{
    activation::Activation, loss::LossFunction, optimizers::Optimizer,
    regularization::Regularization, scheduler::Scheduler,
};
use rand::seq::SliceRandom;

pub struct GradientDescentSolver {
    pub scheduler: Scheduler,
    pub optimizer: Optimizer,
    pub activation: Activation,
    pub loss: LossFunction,
}
impl GradientDescentSolver {
    pub fn train(
        &mut self,
        data: &DMatrix<f64>,
        targets: &DMatrix<f64>,
        epochs: usize,
        learning_rate: f64,
        n_batches: usize,
        silent: bool,
        regularizer: &Regularization,
    ) -> DMatrix<f64> {
        let mut rng = rand::thread_rng();
        let mut eta: f64;

        let mut weights = DMatrix::from_element(data.ncols(), targets.ncols(), 1.0);
        let batch_size = if n_batches == 0 {
            1
        } else {
            data.nrows() / n_batches
        };

        for epoch in 0..epochs {
            let mut order: Vec<usize> = (0..n_batches).collect();
            order.shuffle(&mut rng);
            for j in order {
                let remaining = data.nrows() - (j * batch_size);
                let current_batch_size = if remaining < batch_size {
                    remaining 
                } else {
                    batch_size
                };
                let batch_data = data.rows(j * batch_size, current_batch_size);
                let h: DMatrix<f64> = self.activation.call_on_all(batch_data * &weights);
                let y: DMatrix<f64> = targets.rows(j * batch_size, current_batch_size).map(|x| x);
                let errors = self.loss.loss_d(&y, &h);
                eta = self.scheduler.eta(learning_rate, epoch);
                let gradient = &batch_data.transpose() * &errors;
                let coeff = regularizer.coeff(&weights);
                self.optimizer.optimize(&mut weights, gradient, eta, coeff);
            }
            if !silent {
                let h = self.activation.call_on_all(data * &weights);
                let error: f64 = self.loss.loss(&targets, &h).sum() / targets.len() as f64;
                println!("Epoch <{}: Current Errors {}", epoch, error);
            }
        }
        weights
    }
    pub fn predict(&self, data: &DMatrix<f64>, weights: &DMatrix<f64>) -> DMatrix<f64> {
        let res = data * weights;
        self.activation.call_on_all(res)
    }
}
