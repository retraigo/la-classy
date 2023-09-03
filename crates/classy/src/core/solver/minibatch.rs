use nalgebra::{DMatrix, DVector};

use crate::core::{
    activation::Activation, loss::LossFunction, optimizers::Optimizer,
    regularization::Regularization, scheduler::Scheduler,
};
use rand::{Rng, seq::SliceRandom};

pub struct MinibatchSGDSolver {
    pub scheduler: Scheduler,
    pub optimizer: Optimizer,
    pub activation: Activation,
    pub loss: LossFunction,
}
impl MinibatchSGDSolver {
    pub fn train(
        &mut self,
        data: &DMatrix<f64>,
        targets: &DVector<f64>,
        epochs: usize,
        learning_rate: f64,
        n_batches: usize,
        silent: bool,
        regularizer: &Regularization,
    ) -> DVector<f64> {
        let mut rng = rand::thread_rng();
        let mut eta = learning_rate;

        let mut weights = DVector::from_element(data.ncols(), 1.0);
        let batch_size = data.nrows() / n_batches;

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
                let inverse_batch_size = 1.0 / current_batch_size as f64;
                let batch_data = data.rows(j * batch_size, current_batch_size);
                let h = self.activation.call_on_all(batch_data * &weights);
                let y: DVector<f64> = targets.rows(j * batch_size, current_batch_size).map(|x| x);
                let errors = self.loss.loss_d(&y, &h);
                eta = self.scheduler.eta(learning_rate, epoch);
                let gradient = &batch_data.transpose() * &errors * inverse_batch_size;
                let coeff = regularizer.coeff(&weights);

                self.optimizer.optimize(&mut weights, gradient, eta, coeff);
            }
            if epoch % 100 == 0 && !silent {
                let h = self.activation.call_on_all(data * &weights);
                let error: f64 = self.loss.loss(&targets, &h).sum() / targets.len() as f64;
                println!("Epoch <{}: Current Errors {}", epoch, error);
            }
        }
        weights
    }
}
