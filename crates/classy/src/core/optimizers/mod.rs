use ndarray::Array2;
mod adam;
mod rmsprop;

pub enum OptimizerConfig {
    None,
    Adam {
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
    RMSProp {
        decay_rate: f32,
        epsilon: f32,
    },
}
pub enum Optimizer {
    None,
    Adam(adam::AdamOptimizer),
    RMSProp(rmsprop::RMSPropOptimizer),
}

impl Optimizer {
    pub fn from(config: OptimizerConfig, input_size: usize, output_size: usize) -> Self {
        match config {
            OptimizerConfig::None => Self::None,
            OptimizerConfig::Adam {
                beta1,
                beta2,
                epsilon,
            } => Self::Adam(adam::AdamOptimizer::new(
                beta1,
                beta2,
                epsilon,
                input_size,
                output_size,
            )),
            OptimizerConfig::RMSProp {
                decay_rate,
                epsilon,
            } => Self::RMSProp(rmsprop::RMSPropOptimizer::new(
                decay_rate,
                epsilon,
                input_size,
                output_size,
            )),
        }
    }
    pub fn optimize(
        &mut self,
        weights: &mut Array2<f32>,
        gradient: &Array2<f32>,
        learning_rate: f32,
        regularization: &Array2<f32>,
    ) {
        match self {
            Self::Adam(adam) => adam.optimize(weights, gradient, learning_rate, regularization),
            Self::RMSProp(rmsprop) => rmsprop.optimize(weights, gradient, learning_rate, regularization),
            Self::None => {
                *weights = weights.clone() - (gradient + regularization) * learning_rate;
            }
        }
    }
}
