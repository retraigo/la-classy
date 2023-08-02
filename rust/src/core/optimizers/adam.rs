use nalgebra::DVector;

use crate::types::AdamOptimizerConfig;

#[derive(Debug)]
pub struct AdamOptimizer {
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: DVector<f64>,
    v: DVector<f64>,
    t: usize,
}

impl AdamOptimizer {
    pub fn new(config: AdamOptimizerConfig, n_weights: usize) -> AdamOptimizer {
        let m = DVector::zeros(n_weights);
        let v = DVector::zeros(n_weights);
        AdamOptimizer {
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            m,
            v,
            t: 1,
        }
    }
    pub fn optimize(
        &mut self,
        weights: &mut DVector<f64>,
        gradient: DVector<f64>,
        learning_rate: f64,
        l1: DVector<f64>,
    ) {
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * (&gradient + &l1);
        self.v = self.beta2 * &self.v
            + (1.0 - self.beta2) * (&gradient.component_mul(&gradient) + l1.component_mul(&l1));

        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));
        *weights -= &m_hat.zip_map(&v_hat.map(|x| x.sqrt()).add_scalar(self.epsilon), |x, y| {
            x / y
        }) * learning_rate;
        self.t += 1;
    }
}
