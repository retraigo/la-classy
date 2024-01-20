use nalgebra::DMatrix;

#[derive(Debug)]
pub struct RMSPropOptimizer {
    decay_rate: f64,
    epsilon: f64,
    acc_sg: DMatrix<f64>,
}

impl RMSPropOptimizer {
    pub fn new(
        decay_rate: f64,
        epsilon: f64,
        input_size: usize,
        output_size: usize,
    ) -> RMSPropOptimizer {
        let acc: DMatrix<f64> = DMatrix::zeros(input_size, output_size);
        RMSPropOptimizer {
            decay_rate,
            epsilon,
            acc_sg: acc,
        }
    }
    pub fn optimize(
        &mut self,
        weights: &mut DMatrix<f64>,
        gradient: DMatrix<f64>,
        learning_rate: f64,
        l: DMatrix<f64>,
    ) {
        self.acc_sg *= self.decay_rate;
        self.acc_sg += (1.0 - self.decay_rate) * gradient.component_mul(&gradient);

        // Update parameters using RMSprop update rule
        *weights -= learning_rate
            * (gradient.component_div(&((self.acc_sg.add_scalar(self.epsilon)).map(|x| x.sqrt())))
                - l);
    }
}
