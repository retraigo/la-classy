use std::ops::Add;

use ndarray::Array1;

#[derive(Debug)]
pub struct AdamOptimizer {
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Array1<f64>,
    v: Array1<f64>,
    t: usize,
}

impl AdamOptimizer {
    pub fn new(beta1: f64, beta2: f64, epsilon: f64, input_size: usize) -> AdamOptimizer {
        let m = Array1::zeros(input_size);
        let v = Array1::zeros(input_size);
        AdamOptimizer {
            beta1,
            beta2,
            epsilon,
            m,
            v,
            t: 1,
        }
    }
    pub fn optimize(
        &mut self,
        weights: &mut Array1<f64>,
        gradient: Array1<f64>,
        learning_rate: f64,
        l: Array1<f64>,
    ) {
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * (&gradient + &l);
        self.v = self.beta2 * &self.v
            + (1.0 - self.beta2) * (&gradient* &gradient + l.clone() * &l);

        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));
        let subm: Array1<f64> = m_hat.iter().zip(
            &v_hat
            .map(|x| x.sqrt())
            .add(self.epsilon)
        ).map(|(x, y)| {
            x / y
        }).collect();
        *weights = weights.clone() - (subm * learning_rate);
        self.t += 1;
    }
}
