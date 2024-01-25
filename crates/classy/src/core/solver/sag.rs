use nalgebra::DMatrix;

use crate::core::{
    activation::Activation, loss::LossFunction, optimizers::Optimizer,
    regularization::Regularization, scheduler::Scheduler,
};
pub struct SAGSolver {
    pub scheduler: Scheduler,
    pub optimizer: Optimizer,
    pub activation: Activation,
    pub loss: LossFunction,
}
impl SAGSolver {
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
        let mut eta: f64;

        let mut weights = DMatrix::from_element(data.ncols(), targets.ncols(), 1.0);
        let mut gradients =
            vec![DMatrix::from_element(data.ncols(), targets.ncols(), 0.0); data.nrows()];
        let mut average_gradients = DMatrix::from_element(data.ncols(), targets.ncols(), 0.0);
        let batch_size = if n_batches == 0 {
            1
        } else {
            data.nrows() / n_batches
        };

        for epoch in 0..epochs {
            for batch_start in (0..data.nrows()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(data.nrows());

                // Calculate the average gradient for the minibatch
                for i in batch_start..batch_end {
                    let current_data: DMatrix<f64> = data.rows(i, 1).map(|x| x);
                    let h: DMatrix<f64> = self.predict(&current_data, &weights);
                    let y: DMatrix<f64> = targets.rows(i, 1).map(|x| x);
                    let errors = self.loss.loss_d(&y, &h);
                    let gradient = data.row(i).transpose() * errors;
                    average_gradients -= gradients[i].clone();
                    average_gradients += &gradient;
                    gradients[i] = gradient.clone();
                }
                let coeff = regularizer.coeff(&weights);
                eta = self.scheduler.eta(learning_rate, epoch);
                self.optimizer.optimize(
                    &mut weights,
                    &average_gradients / batch_size as f64,
                    eta,
                    coeff,
                );
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
