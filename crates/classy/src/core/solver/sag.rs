// TODO

use ndarray::{Array2, Array1};

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

        let mut average_gradients: Vec<Array1<f64>> = vec![Array1::zeros(data.ncols()); data.nrows()];
        let mut prev_gradients: Vec<Array1<f64>> = vec![Array1::zeros(data.ncols()); data.nrows()];

        for epoch in 0..epochs {
            let mut total_gradients: Array1<f64> = Array1::zeros(data.ncols());
            for _ in 0..data.nrows() {
                let j = rng.gen_range(0..data.nrows());
                let current_data = data.row(j);

                let h =
                    Array1::from_vec(vec![self.activation.call(current_data.dot(&weights))]);

                let error = self.loss.loss_d(
                    &Array1::from_vec(vec![*targets.get(j).unwrap_or(&0.0)]),
                    &h,
                );
                eta = self.scheduler.eta(learning_rate, epoch);
                let gradient = &current_data.t().dot(&error) * inverse_n;

                total_gradients += &gradient - &prev_gradients[j] + &average_gradients[j];

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
