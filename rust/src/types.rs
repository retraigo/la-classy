use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub epochs: usize,
    pub silent: bool,
    pub fit_intercept: bool,
    pub model: Model,
    pub loss: LossFunction,
    pub optimizer: Optimizer,
    pub c: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum LossFunction {
    MAE,
    MSE,
    CrossEntropy,
    BinCrossEntropy,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Model {
    None,
    Logit,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Optimizer {
    Adam(AdamOptimizerConfig),
    SGD,
    MinibatchSGD(MinibatchSGDOptimizerConfig),
    GD,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AdamOptimizerConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub n_batches: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MinibatchSGDOptimizerConfig {
    pub n_batches: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Scheduler {
    None,
    ExponentialAnnealer {
        // Rate of decay
        rate: f64,
    },
    LinearAnnealer {
        // Rate of decay
        rate: f64,
    },
    DecayScheduler {
        // Rate of decay
        rate: f64,
        // Number of epochs for decay
        step_size: usize,
    },
    OneCycleScheduler {
        // Max allowed learning rate
        max_lr: f64,
        // Number of steps in one cycle
        cycle_steps: usize,
    },
}
