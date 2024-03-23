use ndarray::{s, Array2};

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
        data: &Array2<f64>,
        targets: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
        n_batches: usize,
        silent: bool,
        tolerance: f64,
        patience: isize,
        regularizer: &Regularization,
    ) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut eta: f64;

        let mut weights = Array2::from_elem((data.ncols(), targets.ncols()), 1.0);

        let mut best_weights = weights.clone();
        let mut best_loss = f64::INFINITY;
        let mut disappointment = 0;

        let mut previous_weights = weights.clone();
        let batch_size = if n_batches == 0 {
            1
        } else {
            data.nrows() / n_batches
        };

        'iters: for epoch in 0..epochs {
            let mut order: Vec<usize> = (0..n_batches).collect();
            order.shuffle(&mut rng);
            for j in order {
                let remaining = data.nrows() - (j * batch_size);
                let current_batch_size = if remaining < batch_size {
                    remaining
                } else {
                    batch_size
                };
                let start = j * batch_size;
                let batch_data = data.slice(s![start..(start + current_batch_size), ..]);
                let h: Array2<f64> = self.activation.call_on_all(batch_data.dot(&weights));
                let y: Array2<f64> = targets.slice(s![start..(start + current_batch_size), ..]).clone().map(|x| *x);
                let errors = self.loss.loss_d(&y, &h);
                eta = self.scheduler.eta(learning_rate, epoch);
                let gradient = &batch_data.t().dot(&errors);
                let coeff = regularizer.coeff(&weights);
                self.optimizer.optimize(&mut weights, gradient.clone(), eta, coeff);
                if tolerance > 0.0 {
                    let difference = (&weights - &previous_weights)
                        .map(|x| x.powi(2))
                        .sum()
                        .sqrt();
                    if difference < tolerance {
                        println!("Converged in {} epochs.", epoch);
                        break 'iters;
                    } else {
                        previous_weights = weights.clone()
                    }
                }
            }
            if !silent || patience != -1 {
                let h = self.activation.call_on_all(data.dot(&weights));
                let error: f64 = self.loss.loss(&targets, &h).sum() / targets.len() as f64;
                if patience != -1 {
                    if error < best_loss {
                        disappointment = 0;
                        best_loss = error;
                        best_weights = weights.clone()
                    } else {
                        disappointment += 1;
                        if disappointment >= patience {
                            println!("Stopping early because there has been no improvement for {} epochs.", disappointment);
                            weights = best_weights.clone();
                            break;
                        }
                    }
                }
                if !silent {
                    println!("Epoch <{}: Current Errors {}", epoch, error);
                }
            }
        }
        weights
    }
    pub fn predict(&self, data: &Array2<f64>, weights: &Array2<f64>) -> Array2<f64> {
        let res = data.dot(weights);
        self.activation.call_on_all(res)
    }
}
