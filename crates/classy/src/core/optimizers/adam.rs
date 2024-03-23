use ndarray::{Array1, Array2};

#[derive(Debug)]
pub struct AdamOptimizer {
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Array2<f64>,
    v: Array2<f64>,
    t: usize,
}

impl AdamOptimizer {
    pub fn new(
        beta1: f64,
        beta2: f64,
        epsilon: f64,
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
        weights: &mut Array2<f64>,
        gradient: Array2<f64>,
        learning_rate: f64,
        l: Array2<f64>,
    ) {
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * (&gradient + &l);
        self.v = self.beta2 * &self.v + (1.0 - self.beta2) * (&gradient * (&gradient) + &l * (&l));

        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));
        *weights = weights.clone()
            - (&m_hat
                .iter()
                .zip(&v_hat.map(|x| x.sqrt()) + (self.epsilon))
                .map(|(x, y)| x / y)
                .collect::<Array1<f64>>()
                .to_shape((weights.nrows(), weights.ncols()))
                .unwrap()
                .to_owned()
                * learning_rate);
        self.t += 1;
    }
}
