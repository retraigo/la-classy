use ndarray::{Array2, Array1};

use crate::core::{
    activation::Activation, loss::LossFunction, optimizers::Optimizer,
    regularization::Regularization, scheduler::Scheduler,
};
use rand::{Rng, seq::SliceRandom};

pub struct SGDSolver {
    pub scheduler: Scheduler,
    pub optimizer: Optimizer,
    pub activation: Activation,
    pub loss: LossFunction,
}
impl SGDSolver {
    pub fn train(
        &mut self,
        data: &Array2<f64>,
        targets: &Array1<f64>,
        epochs: usize,
        learning_rate: f64,
        silent: bool,
        regularizer: &Regularization,
    ) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let mut eta = learning_rate;
        let inverse_n = 1.0 / data.nrows() as f64;

        let mut weights = Array1::from_elem(data.ncols(), 1.0);

        for epoch in 0..epochs {
            let mut order: Vec<usize> = (0..data.nrows()).collect();
            order.shuffle(&mut rng);
            for j in order {
                let current_data = data.row(j);

                let h =
                Array1::from_vec(vec![self.activation.call(current_data.t().dot(&weights))]);

                let error = self.loss.loss_d(
                    &Array1::from_vec(vec![*targets.get(j).unwrap_or(&0.0)]),
                    &h,
                );
                eta = self.scheduler.eta(learning_rate, epoch);
                let gradient = &current_data * &error * inverse_n;
                let coeff = regularizer.coeff(&weights);

                self.optimizer.optimize(&mut weights, gradient, eta, coeff);
            }
            if epoch % 100 == 0 && !silent {
                let h = self.activation.call_on_all(data.dot(&weights));
                let error: f64 = self.loss.loss(&targets, &h).sum() / targets.len() as f64;
                println!("Epoch <{}: Current Errors {}", epoch, error);
            }
        }
        weights
    }
}
