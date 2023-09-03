use nalgebra::DVector;
mod adam;

pub enum OptimizerConfig {
    None,
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
}
pub enum Optimizer {
    None,
    Adam(adam::AdamOptimizer),
}

impl Optimizer {
    pub fn from(config: OptimizerConfig, input_size: usize) -> Self {
        match config {
            OptimizerConfig::None => Self::None,
            OptimizerConfig::Adam {
                beta1,
                beta2,
                epsilon,
            } => Self::Adam(adam::AdamOptimizer::new(beta1, beta2, epsilon, input_size)),
        }
    }
    pub fn optimize(
        &mut self,
        weights: &mut DVector<f64>,
        gradient: DVector<f64>,
        learning_rate: f64,
        regularization: DVector<f64>,
    ) {
        match self {
            Self::Adam(adam) => adam.optimize(weights, gradient, learning_rate, regularization),
            Self::None => {
                *weights -= (gradient + regularization) * learning_rate;
            }
        }
    }
}
