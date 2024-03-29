use nalgebra::DMatrix;

#[derive(Debug)]
pub struct AdamOptimizer {
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: DMatrix<f64>,
    v: DMatrix<f64>,
    t: usize,
}

impl AdamOptimizer {
    pub fn new(beta1: f64, beta2: f64, epsilon: f64, input_size: usize, output_size: usize) -> AdamOptimizer {
        let m = DMatrix::zeros(input_size, output_size);
        let v = DMatrix::zeros(input_size, output_size);
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
        weights: &mut DMatrix<f64>,
        gradient: DMatrix<f64>,
        learning_rate: f64,
        l: DMatrix<f64>,
    ) {
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * (&gradient + &l);
        self.v = self.beta2 * &self.v
            + (1.0 - self.beta2) * (&gradient.component_mul(&gradient) + l.component_mul(&l));

        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));
        *weights -= &m_hat.zip_map(&v_hat.map(|x| x.sqrt()).add_scalar(self.epsilon), |x, y| {
            x / y
        }) * learning_rate;
        self.t += 1;
    }
}
