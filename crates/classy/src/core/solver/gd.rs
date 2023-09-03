use nalgebra::{DMatrix, DVector};

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
        data: &DMatrix<f64>,
        targets: &DVector<f64>,
        epochs: usize,
        learning_rate: f64,
        silent: bool,
        regularizer: &Regularization,
    ) -> DVector<f64> {
        let mut eta = learning_rate;
        let inverse_n = 1.0 / data.nrows() as f64;
        let mut weights = DVector::from_element(data.ncols(), 1.0);
     //   println!("I CAME");

        for epoch in 0..epochs {
            let h = self.activation.call_on_all(data * &weights);
            
            let errors = self.loss.loss_d(&targets, &h);
            eta = self.scheduler.eta(learning_rate, epoch);
            let gradient = &data.transpose() * &errors * inverse_n;
            let coeff = regularizer.coeff(&weights);
            //println!("H {:?} \nS {:?} \nG {:?}", weights.as_slice(), errors.as_slice(), coeff.as_slice());
            self.optimizer.optimize(&mut weights, gradient, eta, coeff);

            if epoch % 100 == 0 && !silent {
                let error: f64 = self.loss.loss(&targets, &h).sum() / targets.len() as f64;
                println!("Epoch <{}: Current Errors {}", epoch, error);
            }
        }
        weights
    }
}
