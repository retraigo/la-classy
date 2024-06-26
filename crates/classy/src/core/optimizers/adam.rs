use ndarray::Array2;
use std::ops::{SubAssign, Mul, Div, Add};

#[derive(Debug)]
pub struct AdamOptimizer {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: Array2<f32>,
    v: Array2<f32>,
    t: i32,
}

impl AdamOptimizer {
    pub fn new(
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        input_size: usize,
        output_size: usize,
    ) -> AdamOptimizer {
        let m = Array2::zeros((input_size, output_size));
        let v = Array2::zeros((input_size, output_size));
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
        weights: &mut Array2<f32>,
        gradient: &Array2<f32>,
        learning_rate: f32,
        l: &Array2<f32>,
    ) {
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * (gradient + l);
        self.v = self.beta2 * &self.v + (1.0 - self.beta2) * (gradient.mul(gradient) + l * (l));

        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t));
        weights.sub_assign(&learning_rate.mul(m_hat).div(v_hat.map(|x| x.sqrt()).add(self.epsilon)));
        self.t += 1;
    }
}
