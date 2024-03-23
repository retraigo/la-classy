use ndarray::Array2;

#[derive(Debug)]
pub struct RMSPropOptimizer {
    decay_rate: f32,
    epsilon: f32,
    acc_sg: Array2<f32>,
}

impl RMSPropOptimizer {
    pub fn new(
        decay_rate: f32,
        epsilon: f32,
        input_size: usize,
        output_size: usize,
    ) -> RMSPropOptimizer {
        let acc: Array2<f32> = Array2::zeros((input_size, output_size));
        RMSPropOptimizer {
            decay_rate,
            epsilon,
            acc_sg: acc,
        }
    }
    pub fn optimize(
        &mut self,
        weights: &mut Array2<f32>,
        gradient: &Array2<f32>,
        learning_rate: f32,
        l: &Array2<f32>,
    ) {
        self.acc_sg *= self.decay_rate;
        self.acc_sg = &self.acc_sg + (1.0 - self.decay_rate) * gradient * (gradient);

        // Update parameters using RMSprop update rule
        *weights = weights.clone()
            - learning_rate
                * (gradient / (&((&self.acc_sg + (self.epsilon)).map(|x| x.sqrt()))) - l);
    }
}
