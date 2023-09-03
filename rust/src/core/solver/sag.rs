// TODO

use nalgebra::{DMatrix, DVector};

use crate::core::{
    activation::Activation, loss::LossFunction, optimizers::Optimizer,
    regularization::Regularization, scheduler::Scheduler,
};
use rand::Rng;

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
        targets: &DVector<f64>,
        epochs: usize,
        learning_rate: f64,
        silent: bool,
        regularizer: &Regularization,
    ) -> DVector<f64> {
        let mut rng = rand::thread_rng();
        let mut eta = learning_rate;
        let inverse_n = 1.0 / data.nrows() as f64;

        let mut weights = DVector::from_element(data.ncols(), 1.0);

        let mut average_gradients: Vec<DVector<f64>> = vec![DVector::zeros(data.ncols()); data.nrows()];
        let mut prev_gradients: Vec<DVector<f64>> = vec![DVector::zeros(data.ncols()); data.nrows()];

        for epoch in 0..epochs {
            let mut total_gradients: DVector<f64> = DVector::zeros(data.ncols());
            for _ in 0..data.nrows() {
                let j = rng.gen_range(0..data.nrows());
                let current_data = data.row(j);

                let h =
                    DVector::from_vec(vec![self.activation.call(current_data.dot(&weights))]);

                let error = self.loss.loss_d(
                    &DVector::from_vec(vec![*targets.get(j).unwrap_or(&0.0)]),
                    &h,
                );
                eta = self.scheduler.eta(learning_rate, epoch);
                let gradient = &current_data.transpose() * &error * inverse_n;

                total_gradients += &gradient - &prev_gradients[j] + &average_gradients[j];

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
