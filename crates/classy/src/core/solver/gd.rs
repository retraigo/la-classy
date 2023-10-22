use ndarray::{Array2, Array1};
use crate::core::{
    activation::Activation, loss::LossFunction, optimizers::Optimizer,
    regularization::Regularization, scheduler::Scheduler,
};

pub struct GradientDescentSolver {
    pub scheduler: Scheduler,
    pub optimizer: Optimizer,
    pub activation: Activation,
    pub loss: LossFunction,
}
impl GradientDescentSolver {
    pub fn train(
        &mut self,
        data: &Array2<f64>,
        targets: &Array1<f64>,
        epochs: usize,
        learning_rate: f64,
        silent: bool,
        regularizer: &Regularization,
    ) -> Array1<f64> {
        let mut eta = learning_rate;
        let inverse_n = 1.0 / data.nrows() as f64;
        let mut weights = Array1::from_elem(data.ncols(), 1.0);
        for epoch in 0..epochs {
            let h = self.activation.call_on_all(data.dot(&weights));
            
            let errors = self.loss.loss_d(&targets, &h);
            eta = self.scheduler.eta(learning_rate, epoch);
            let gradient = data.t().dot(&errors) * inverse_n;
            let coeff = regularizer.coeff(&weights);
            self.optimizer.optimize(&mut weights, gradient, eta, coeff);

            if epoch % 100 == 0 && !silent {
                let error: f64 = self.loss.loss(&targets, &h).sum() / targets.len() as f64;
                println!("Epoch <{}: Current Errors {}", epoch, error);
            }
        }
        weights
    }
}
